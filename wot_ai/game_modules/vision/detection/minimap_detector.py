#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimap_detector_refactored.py
重构版小地图检测器：
复用 DetectionEngine，实现小地图检测逻辑。
"""

from typing import Optional, Tuple
import numpy as np
from loguru import logger

from wot_ai.vision.detection_engine import DetectionEngine


class MinimapDetector:
    """重构版：依赖 DetectionEngine 实现小地图检测"""

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("model_path不能为空")

        self.engine_ = DetectionEngine(model_path)
        self.conf_threshold_ = 0.25

    def LoadModel(self):
        return self.engine_.LoadModel()

    def Detect(self, frame_minimap: np.ndarray, confidence_threshold: Optional[float] = None) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """检测小地图元素（直接输入小地图图像）
        返回 self_pos, flag_pos
        """
        if frame_minimap is None or frame_minimap.size == 0:
            logger.error("frame_minimap为空或无效")
            return None, None

        detections = self.engine_.Detect(frame_minimap, confidence_threshold or self.conf_threshold_)
        self_pos, flag_pos = None, None

        for d in detections:
            cls, bbox = d['class'], d['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            if cls == 0:  # 我方
                self_pos = (cx, cy)
            elif cls == 3:  # 敌方基地
                flag_pos = (cx, cy)

        return self_pos, flag_pos

    def Reset(self):
        pass