#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测模块（重构版）

支持 YOLO / 传统方案 + 异步检测 + 结果缓存
"""

import cv2
import numpy as np
import threading
import time
from typing import Dict, Optional, List
from pathlib import Path
from ultralytics import YOLO

# 统一导入机制
from wot_ai.utils.paths import setup_python_path, resolve_path
from wot_ai.utils.imports import try_import_multiple, import_function
setup_python_path()

# 尝试多种导入方式
PositionSmoother = import_function([
    'wot_ai.game_modules.navigation.core.position_smoother',
    'game_modules.navigation.core.position_smoother',
    'navigation.core.position_smoother'
], 'PositionSmoother')
if PositionSmoother is None:
    from .position_smoother import PositionSmoother

SetupLogger = None
logger_module, _ = try_import_multiple([
    'wot_ai.game_modules.common.utils.logger',
    'game_modules.common.utils.logger',
    'common.utils.logger',
    'yolo.utils.logger'
])
if logger_module is not None:
    SetupLogger = getattr(logger_module, 'SetupLogger', None)

if SetupLogger is None:
    from ...common.utils.logger import SetupLogger

logger = SetupLogger(__name__)


# =============================
# 通用接口定义
# =============================
class BaseMinimapDetector:
    """抽象基类，定义统一接口"""
    def Detect(self, frame: np.ndarray) -> Dict:
        raise NotImplementedError("Detect() must be implemented.")


# =============================
# YOLO 方案
# =============================
class YOLODetector(BaseMinimapDetector):
    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        self.model_path_ = model_path
        self.confidence_threshold_ = confidence_threshold
        self.model_ = None
        self.self_pos_smoother_ = PositionSmoother(window=3)
        self.flag_pos_smoother_ = PositionSmoother(window=3)

    def LoadModel(self) -> bool:
        """加载 YOLO 模型"""
        try:
            model_path_obj = Path(self.model_path_)
            if not model_path_obj.exists():
                logger.error(f"模型文件不存在: {self.model_path_}")
                return False
            
            logger.info(f"加载小地图检测模型: {model_path_obj.absolute()}")
            self.model_ = YOLO(str(model_path_obj))
            logger.info("小地图检测模型加载成功")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def Detect(self, frame: np.ndarray) -> Dict:
        if self.model_ is None:
            logger.error("模型未加载，请先调用 LoadModel()")
            return self._Empty()
        
        try:
            results = self.model_(frame, conf=self.confidence_threshold_, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    detections.append((cls, conf, xyxy))
            return self._ParseDetections(detections)
        except Exception as e:
            logger.error(f"YOLO检测失败: {e}")
            return self._Empty()

    def _ParseDetections(self, dets):
        result = {'self_pos': None, 'flag_pos': None, 'obstacles': [], 'roads': []}
        for cls, conf, xyxy in dets:
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            if cls == 0:  # 自己箭头
                result['self_pos'] = self.self_pos_smoother_.Smooth((cx, cy))
            elif cls == 3:  # 基地
                result['flag_pos'] = self.flag_pos_smoother_.Smooth((cx, cy))
            elif cls == 4:  # 障碍
                result['obstacles'].append(xyxy)
            elif cls == 5:  # 道路
                result['roads'].append(xyxy)
        return result

    def _Empty(self):
        return {'self_pos': None, 'flag_pos': None, 'obstacles': [], 'roads': []}

    def ResetSmoothers(self):
        """重置位置平滑器"""
        self.self_pos_smoother_.Reset()
        self.flag_pos_smoother_.Reset()


# =============================
# 传统方案
# =============================
class TraditionalDetector(BaseMinimapDetector):
    """传统方案：颜色+形状检测箭头，HSV检测基地"""
    def __init__(self):
        self.self_pos_smoother_ = PositionSmoother(window=3)

    def Detect(self, frame: np.ndarray) -> Dict:
        result = {'self_pos': None, 'flag_pos': None, 'obstacles': [], 'roads': []}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 检测自己箭头（白色）---
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if 30 < area < 500:
                x, y, w, h = cv2.boundingRect(c)
                result['self_pos'] = self.self_pos_smoother_.Smooth((x + w / 2, y + h / 2))
                break

        # --- 检测基地（红色圆）---
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius > 5:
                result['flag_pos'] = (x, y)
        return result

    def ResetSmoothers(self):
        """重置位置平滑器"""
        self.self_pos_smoother_.Reset()


# =============================
# 管理器：异步 + 模式切换
# =============================
class MinimapManager:
    """封装异步检测与缓存逻辑"""
    def __init__(self, mode: str = "yolo", model_path: Optional[str] = None):
        self.mode_ = mode
        self.model_path_ = model_path
        self.detector_ = None
        self.result_cache_ = {'self_pos': None, 'flag_pos': None, 'obstacles': [], 'roads': []}
        self.last_detect_time_ = 0.0
        self.interval_ = 0.3  # 检测间隔
        self.thread_ = None
        self.stop_flag_ = False
        self._InitDetector()

    def _InitDetector(self):
        if self.mode_ == "yolo":
            if self.model_path_ is None:
                raise ValueError("YOLO模式需要提供model_path")
            self.detector_ = YOLODetector(self.model_path_)
            if not self.detector_.LoadModel():
                raise RuntimeError("YOLO模型加载失败")
        else:
            self.detector_ = TraditionalDetector()

    def SwitchMode(self, mode: str):
        """切换检测模式"""
        logger.info(f"切换检测模式: {self.mode_} -> {mode}")
        self.mode_ = mode
        self._InitDetector()

    def _DetectThread(self, frame: np.ndarray):
        """检测线程函数"""
        self.result_cache_ = self.detector_.Detect(frame)
        self.last_detect_time_ = time.time()

    def Detect(self, frame: np.ndarray) -> Dict:
        """检测接口（异步）"""
        now = time.time()
        if now - self.last_detect_time_ > self.interval_:
            if self.thread_ is None or not self.thread_.is_alive():
                self.thread_ = threading.Thread(target=self._DetectThread, args=(frame,))
                self.thread_.daemon = True
                self.thread_.start()
        return self.result_cache_


# =============================
# 兼容层：保持原有接口
# =============================
class MinimapDetector:
    """小地图检测器（兼容原有接口）"""
    def __init__(self, model_path: str, minimap_region: Dict, base_dir: Optional[Path] = None):
        """
        初始化小地图检测器
        
        Args:
            model_path: YOLO模型路径
            minimap_region: 小地图区域配置 {'x': int, 'y': int, 'width': int, 'height': int}
            base_dir: 基础目录（用于解析相对路径）
        """
        # 使用统一路径解析
        self.model_path_ = str(resolve_path(model_path, base_dir))
        
        self.minimap_region_ = minimap_region
        self.base_dir_ = base_dir
        
        # 使用 YOLO 检测器
        self.detector_ = YOLODetector(self.model_path_)
        self.model_loaded_ = False

    def LoadModel(self) -> bool:
        """加载 YOLO 模型"""
        if self.model_loaded_:
            return True
        success = self.detector_.LoadModel()
        if success:
            self.model_loaded_ = True
        return success

    def ExtractMinimap(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从完整屏幕帧中提取小地图区域
        
        Args:
            frame: 屏幕帧（BGR格式），可能是完整屏幕或已裁剪的小地图区域
        
        Returns:
            小地图区域（BGR格式），如果区域无效则返回 None
        """
        try:
            width = self.minimap_region_['width']
            height = self.minimap_region_['height']
            
            # 如果 frame 的尺寸已经匹配小地图区域，说明已经裁剪过了，直接返回
            if frame.shape[0] == height and frame.shape[1] == width:
                return frame
            
            # 否则，从完整屏幕帧中提取小地图区域
            x = self.minimap_region_['x']
            y = self.minimap_region_['y']
            
            # 检查边界
            if x < 0 or y < 0:
                logger.warning(f"小地图区域坐标无效: ({x}, {y})")
                return None
            
            frame_height, frame_width = frame.shape[:2]
            if x + width > frame_width or y + height > frame_height:
                logger.warning(f"小地图区域超出屏幕范围: frame={frame_width}x{frame_height}, "
                             f"region=({x},{y})+{width}x{height}")
                return None
            
            minimap = frame[y:y+height, x:x+width]
            return minimap
        except Exception as e:
            logger.error(f"提取小地图失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def Detect(self, frame: np.ndarray, confidence_threshold: float = 0.25) -> Dict:
        """
        检测小地图中的元素
        
        Args:
            frame: 完整屏幕帧（BGR格式）
            confidence_threshold: 置信度阈值
        
        Returns:
            检测结果字典
        """
        if not self.model_loaded_:
            logger.error("模型未加载，请先调用 LoadModel()")
            return self._EmptyResult()
        
        # 提取小地图区域
        minimap = self.ExtractMinimap(frame)
        if minimap is None:
            return self._EmptyResult()
        
        # 更新置信度阈值
        self.detector_.confidence_threshold_ = confidence_threshold
        
        # 检测
        return self.detector_.Detect(minimap)

    def ParseDetections(self, detections: List[Dict]) -> Dict:
        """
        解析检测结果，转换为结构化格式（兼容方法）
        
        Args:
            detections: YOLO检测结果列表
        
        Returns:
            结构化检测结果
        """
        # 转换为新格式
        dets = []
        for det in detections:
            dets.append((det['class'], det['confidence'], det['bbox']))
        return self.detector_._ParseDetections(dets)

    def _EmptyResult(self) -> Dict:
        """返回空的检测结果"""
        return {
            'self_pos': None,
            'flag_pos': None,
            'obstacles': [],
            'roads': []
        }

    def ResetSmoothers(self):
        """重置位置平滑器"""
        self.detector_.ResetSmoothers()
