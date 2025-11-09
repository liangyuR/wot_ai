#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimap_detector_simplified_with_heading_logfile.py

小地图检测器（简洁版 + 朝向估计 + 控制台/文件日志）
输出:
    - self_pos: 我方中心位置 (x, y)
    - self_angle: 朝向角度 (度)
    - flag_pos: 敌方基地位置 (x, y)
日志:
    - 控制台打印主要信息
    - 同步写入 logs/minimap_detect_YYYY-MM-DD.log
"""

import cv2
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict
from ultralytics import YOLO

from wot_ai.utils.paths import setup_python_path, resolve_path
setup_python_path()

# 导入平滑器
from wot_ai.utils.imports import import_function

PositionSmoother = import_function([
    'wot_ai.game_modules.navigation.core.position_smoother',
    'game_modules.navigation.core.position_smoother',
    'navigation.core.position_smoother'
], 'PositionSmoother')
if PositionSmoother is None:
    from .position_smoother import PositionSmoother

# 导入角点检测器和掩码提供器
from .corner_detector import detect_minimap_corners
from .map_mask_provider import MapMaskProvider


# =============================
# 日志系统
# =============================
def setup_logger(name: str, log_dir: str = "logs"):
    """配置日志系统，写入文件 + 控制台"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 文件日志
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # 控制台日志
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', "%H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"日志记录已启动: {log_file}")
    return logger


# =============================
# 检测器主体
# =============================
class MinimapDetector:
    """小地图检测器：输出 (self_pos, self_angle, flag_pos)，带文件日志"""

    def __init__(self, model_path: str, minimap_region: dict,
                 base_dir: Optional[Path] = None, log_dir: str = "logs",
                 map_id: Optional[str] = None, maps_dir: str = "data/maps", 
                 inflate_px: int = 4):
        self.model_path_ = str(resolve_path(model_path, base_dir))
        self.minimap_region_ = minimap_region
        self.base_dir_ = base_dir
        self.model_ = None
        self.conf_threshold_ = 0.25

        self.self_pos_smoother_ = PositionSmoother(window=3)
        self.flag_pos_smoother_ = PositionSmoother(window=3)
        self.angle_smoother_ = PositionSmoother(window=5)

        # 初始化日志
        self.logger_ = setup_logger("minimap_detect", log_dir)
        
        # 静态掩码支持
        self.map_id_ = map_id
        if map_id:
            # 解析maps_dir路径（支持相对路径）
            if base_dir and not Path(maps_dir).is_absolute():
                maps_dir = str(Path(base_dir).parent.parent / maps_dir)
            self.mask_provider_ = MapMaskProvider(maps_dir, inflate_px)
            self.logger_.info(f"已启用静态掩码支持: map_id={map_id}, maps_dir={maps_dir}, inflate_px={inflate_px}")
        else:
            self.mask_provider_ = None

    def load_model(self) -> bool:
        """加载 YOLO 模型"""
        try:
            path = Path(self.model_path_)
            if not path.exists():
                self.logger_.error(f"模型不存在: {path}")
                return False
            self.model_ = YOLO(str(path))
            self.logger_.info(f"小地图检测模型加载成功: {path}")
            return True
        except Exception as e:
            self.logger_.error(f"加载模型失败: {e}")
            return False

    def extract_minimap(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """直接取屏幕右下角指定长宽的小地图区域"""
        try:
            w = self.minimap_region_['width']
            h = self.minimap_region_['height']
            fw, fh = frame.shape[1], frame.shape[0]
            if w > fw or h > fh:
                self.logger_.warning("小地图尺寸超出屏幕范围")
                return None
            x = fw - w
            y = fh - h
            return frame[y:y+h, x:x+w]
        except Exception as e:
            self.logger_.error(f"提取小地图失败: {e}")
            return None

    def detect(self, minimap: np.ndarray) -> Tuple[
        Optional[Tuple[float, float]],
        Optional[float],
        Optional[Tuple[float, float]]
    ]:
        """
        检测我方与敌方基地位置与朝向

        Returns:
            (self_pos, self_angle, flag_pos)
        """
        if self.model_ is None:
            self.logger_.error("模型未加载，请先调用 load_model()")
            return None, None, None

        try:
            results = self.model_(minimap, conf=self.conf_threshold_, verbose=False)
            self_pos, flag_pos, angle = None, None, None

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2

                    role = None
                    if cls == 0:  # 自己箭头
                        self_pos = self.self_pos_smoother_.Smooth((cx, cy))
                        angle = self._estimate_angle(minimap, xyxy)
                        if angle is not None:
                            angle = self.angle_smoother_.Smooth((angle, 0))[0]
                        role = "self"
                    elif cls == 3:  # 敌方基地
                        flag_pos = self.flag_pos_smoother_.Smooth((cx, cy))
                        role = "flag"
                    else:
                        role = f"class_{cls}"

                    self.logger_.debug(
                        f"[Detect] label={role:<6} conf={conf:.2f} "
                        f"bbox=({xyxy[0]:.1f},{xyxy[1]:.1f},{xyxy[2]:.1f},{xyxy[3]:.1f})"
                    )

            # 输出最终结果汇总日志
            summary = []
            if self_pos:
                summary.append(f"我方位置={tuple(round(v, 1) for v in self_pos)}")
            if angle is not None:
                summary.append(f"朝向={angle:.1f}°")
            if flag_pos:
                summary.append(f"敌方基地={tuple(round(v, 1) for v in flag_pos)}")

            if summary:
                self.logger_.info(" | ".join(summary))

            return self_pos, angle, flag_pos

        except Exception as e:
            self.logger_.error(f"检测失败: {e}")
            return None, None, None

    def _estimate_angle(self, minimap: np.ndarray, bbox) -> Optional[float]:
        """通过箭头形状估计朝向"""
        x1, y1, x2, y2 = map(int, bbox)
        region = minimap[y1:y2, x1:x2]
        if region.size == 0:
            return None

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return None

        try:
            (_, _), (MA, ma), angle = cv2.fitEllipse(cnt)
            angle = (angle + 90) % 360 - 180
            return angle
        except Exception:
            return None

    def Detect(self, frame: np.ndarray, confidence_threshold: float = None) -> Dict:
        """
        检测小地图元素（返回字典格式，包含obstacle_mask）
        
        Args:
            frame: 完整屏幕帧（BGR格式）
            confidence_threshold: 置信度阈值（可选，默认使用self.conf_threshold_）
        
        Returns:
            检测结果字典，包含：
            - 'self_pos': (x, y) 或 None
            - 'flag_pos': (x, y) 或 None
            - 'angle': 角度（度）或 None
            - 'obstacle_mask': 0/1掩码（如果启用静态掩码）或 None
            - 'obstacles': []（向后兼容，保留空列表）
        """
        if confidence_threshold is not None:
            self.conf_threshold_ = confidence_threshold
        
        # 提取小地图
        minimap = self.extract_minimap(frame)
        if minimap is None:
            return self._EmptyResult()
        
        # 检测位置和角度（使用现有的detect方法）
        self_pos, angle, flag_pos = self.detect(minimap)
        
        # 构建结果字典
        result = {
            'self_pos': self_pos,
            'flag_pos': flag_pos,
            'angle': angle,
            'obstacles': [],  # 向后兼容
            'roads': []
        }
        
        # 如果启用了静态掩码，进行角点检测和对齐
        if self.map_id_ and self.mask_provider_:
            try:
                corners = detect_minimap_corners(minimap)
                obstacle_mask = self.mask_provider_.GetAlignedMask(
                    self.map_id_, minimap, corners
                )
                result['obstacle_mask'] = obstacle_mask
                self.logger_.debug(f"静态掩码已对齐: shape={obstacle_mask.shape}, "
                                 f"障碍像素数={np.sum(obstacle_mask)}")
            except Exception as e:
                self.logger_.warning(f"静态掩码对齐失败: {e}，使用空掩码")
                result['obstacle_mask'] = None
        else:
            result['obstacle_mask'] = None
        
        return result
    
    def _EmptyResult(self) -> Dict:
        """返回空的检测结果"""
        return {
            'self_pos': None,
            'flag_pos': None,
            'angle': None,
            'obstacle_mask': None,
            'obstacles': [],
            'roads': []
        }
    
    def reset(self):
        """重置位置和平滑器"""
        self.self_pos_smoother_.Reset()
        self.flag_pos_smoother_.Reset()
        self.angle_smoother_.Reset()
