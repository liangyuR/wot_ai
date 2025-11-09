#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测器：使用YOLO模型检测小地图元素

实现IDetector接口，提供小地图元素检测功能，包括：
- 我方位置检测
- 朝向角度估计
- 敌方基地位置检测
- 静态障碍物掩码支持
"""

# 标准库导入
from pathlib import Path
from typing import Optional, Tuple

# 第三方库导入
import cv2
import numpy as np
from ultralytics import YOLO

# 本地模块导入
from wot_ai.utils.paths import resolve_path
from ..common.imports import GetLogger, ImportNavigationModule
from ..common.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INFLATE_PX,
    DEFAULT_MAPS_DIR,
    SMOOTHER_WINDOW_SELF_POS,
    SMOOTHER_WINDOW_FLAG_POS,
    SMOOTHER_WINDOW_ANGLE
)
from ..common.exceptions import ModelLoadError, DetectionError
from .interfaces import IDetector, DetectionResult

# 导入平滑器
PositionSmoother = ImportNavigationModule('core.position_smoother', 'PositionSmoother')
if PositionSmoother is None:
    from .position_smoother import PositionSmoother

# 导入角点检测器和掩码提供器
from .corner_detector import detect_minimap_corners
from .map_mask_provider import MapMaskProvider

# 统一日志系统
logger = GetLogger()(__name__)


class MinimapDetector(IDetector):
    """
    小地图检测器：使用YOLO模型检测小地图元素
    
    实现IDetector接口，提供小地图元素检测功能，包括：
    - 我方位置检测
    - 朝向角度估计
    - 敌方基地位置检测
    - 静态障碍物掩码支持
    
    示例:
        ```python
        detector = MinimapDetector(
            model_path="path/to/model.pt",
            minimap_region={'x': 1600, 'y': 800, 'width': 320, 'height': 320}
        )
        detector.LoadModel()
        result = detector.Detect(frame)
        ```
    """

    def __init__(self, model_path: str, minimap_region: dict,
                 base_dir: Optional[Path] = None,
                 map_id: Optional[str] = None, maps_dir: str = DEFAULT_MAPS_DIR, 
                 inflate_px: int = DEFAULT_INFLATE_PX):
        # 参数验证
        if not model_path:
            raise ValueError("model_path不能为空")
        if not minimap_region or not isinstance(minimap_region, dict):
            raise ValueError("minimap_region必须是有效的字典")
        required_keys = ['x', 'y', 'width', 'height']
        for key in required_keys:
            if key not in minimap_region:
                raise ValueError(f"minimap_region缺少必需字段: {key}")
            if not isinstance(minimap_region[key], int) or minimap_region[key] < 0:
                raise ValueError(f"minimap_region.{key}必须是正整数")
        if inflate_px < 0:
            raise ValueError("inflate_px必须大于等于0")
        
        self.model_path_ = str(resolve_path(model_path, base_dir))
        self.minimap_region_ = minimap_region
        self.base_dir_ = base_dir
        self.model_ = None
        self.conf_threshold_ = DEFAULT_CONFIDENCE_THRESHOLD

        self.self_pos_smoother_ = PositionSmoother(window=SMOOTHER_WINDOW_SELF_POS)
        self.flag_pos_smoother_ = PositionSmoother(window=SMOOTHER_WINDOW_FLAG_POS)
        self.angle_smoother_ = PositionSmoother(window=SMOOTHER_WINDOW_ANGLE)
        
        # 静态掩码支持
        self.map_id_ = map_id
        if map_id:
            # 解析maps_dir路径（支持相对路径）
            if base_dir and not Path(maps_dir).is_absolute():
                maps_dir = str(Path(base_dir).parent.parent / maps_dir)
            self.mask_provider_ = MapMaskProvider(maps_dir, inflate_px)
            logger.info(f"已启用静态掩码支持: map_id={map_id}, maps_dir={maps_dir}, inflate_px={inflate_px}")
        else:
            self.mask_provider_ = None

    def LoadModel(self) -> bool:
        """加载 YOLO 模型"""
        try:
            path = Path(self.model_path_)
            if not path.exists():
                error_msg = f"模型文件不存在: {path}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg)
            self.model_ = YOLO(str(path))
            logger.info(f"小地图检测模型加载成功: {path}")
            return True
        except ModelLoadError:
            raise
        except Exception as e:
            error_msg = f"加载模型失败: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def ExtractMinimap(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        直接取屏幕右下角指定长宽的小地图区域
        
        Args:
            frame: 屏幕帧（BGR格式）
        
        Returns:
            小地图图像，失败返回None
        
        Raises:
            ValueError: 输入参数无效
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame不能为空")
        if len(frame.shape) < 2:
            raise ValueError("frame必须是有效的图像数组")
        
        try:
            w = self.minimap_region_['width']
            h = self.minimap_region_['height']
            fw, fh = frame.shape[1], frame.shape[0]
            if w > fw or h > fh:
                logger.warning(f"小地图尺寸超出屏幕范围: minimap=({w}, {h}), screen=({fw}, {fh})")
                return None
            x = fw - w
            y = fh - h
            return frame[y:y+h, x:x+w]
        except Exception as e:
            logger.error(f"提取小地图失败: {e}")
            return None

    def DetectMinimap_(self, minimap: np.ndarray) -> Tuple[
        Optional[Tuple[float, float]],
        Optional[float],
        Optional[Tuple[float, float]]
    ]:
        """
        检测我方与敌方基地位置与朝向（内部方法）

        Returns:
            (self_pos, self_angle, flag_pos)
        """
        if self.model_ is None:
            logger.error("模型未加载，请先调用 LoadModel()")
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
                        angle = self.EstimateAngle_(minimap, xyxy)
                        if angle is not None:
                            angle = self.angle_smoother_.Smooth((angle, 0))[0]
                        role = "self"
                    elif cls == 3:  # 敌方基地
                        flag_pos = self.flag_pos_smoother_.Smooth((cx, cy))
                        role = "flag"
                    else:
                        role = f"class_{cls}"

                    logger.debug(
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
                logger.info(" | ".join(summary))

            return self_pos, angle, flag_pos

        except Exception as e:
            error_msg = f"检测失败: {e}"
            logger.error(error_msg)
            raise DetectionError(error_msg) from e

    def EstimateAngle_(self, minimap: np.ndarray, bbox) -> Optional[float]:
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

    def Detect(self, frame: np.ndarray, confidence_threshold: Optional[float] = None) -> DetectionResult:
        """
        检测小地图元素（返回字典格式，包含obstacle_mask）
        
        Args:
            frame: 完整屏幕帧（BGR格式）
            confidence_threshold: 置信度阈值（可选，默认使用self.conf_threshold_）
        
        Returns:
            检测结果字典
        
        Raises:
            ValueError: 输入参数无效
            DetectionError: 检测失败
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame不能为空")
        if confidence_threshold is not None:
            if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
                raise ValueError("confidence_threshold必须在0-1之间")
            self.conf_threshold_ = confidence_threshold
        
        # 提取小地图
        minimap = self.ExtractMinimap(frame)
        if minimap is None:
            return self.EmptyResult_()
        
        # 检测位置和角度（使用现有的DetectMinimap_方法）
        self_pos, angle, flag_pos = self.DetectMinimap_(minimap)
        
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
                logger.debug(f"静态掩码已对齐: shape={obstacle_mask.shape}, "
                                 f"障碍像素数={np.sum(obstacle_mask)}")
            except Exception as e:
                logger.warning(f"静态掩码对齐失败: {e}，使用空掩码")
                result['obstacle_mask'] = None
        else:
            result['obstacle_mask'] = None
        
        return result
    
    def EmptyResult_(self) -> DetectionResult:
        """返回空的检测结果"""
        return {
            'self_pos': None,
            'flag_pos': None,
            'angle': None,
            'obstacle_mask': None,
            'obstacles': [],
            'roads': []
        }
    
    def Reset(self):
        """重置位置和平滑器"""
        self.self_pos_smoother_.Reset()
        self.flag_pos_smoother_.Reset()
        self.angle_smoother_.Reset()
