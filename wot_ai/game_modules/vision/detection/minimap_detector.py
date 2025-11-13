#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重写后的 MinimapDetector：
- 使用全新的 DetectionEngine
- 使用结构化 Detection
- 统一 class ID 常量
- 返回结构更清晰（位置 + 原始检测）
- 更易扩展后续的“朝向 + 障碍物 + 其他元素”
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
from loguru import logger

from ..detection_engine import DetectionEngine, Detection


# ---------------------------------------------------------------
# 小地图类别定义（避免 magic number）
# ---------------------------------------------------------------
CLS_SELF = 0            # 自己的位置图标（小白箭头中心）
CLS_ALLY_FLAG = 1       # 友方基地
CLS_MINI_MAP = 2        # 小地图(弃用)
CLS_ENEMY_FLAG = 3      # 敌方基地


@dataclass
class MinimapDetectionResult:
    """MinimapDetector 的统一输出结构"""
    self_pos: Optional[Tuple[float, float]]
    enemy_flag_pos: Optional[Tuple[float, float]]
    raw_detections: List[Detection]


class MinimapDetector:
    """
    重新设计后的小地图检测器

    输入：一张完整的小地图图像 (frame_minimap)
    输出：self_pos, enemy_flag_pos 以及原始 YOLO detection 列表
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        if not model_path:
            raise ValueError("model_path 不能为空")

        self.engine_ = DetectionEngine(model_path)
        self.conf_threshold_ = conf_threshold

    # -----------------------------------------------------------
    def LoadModel(self) -> bool:
        return self.engine_.LoadModel()

    # -----------------------------------------------------------
    def Detect(self, frame_minimap: np.ndarray, confidence_threshold: Optional[float] = None) -> MinimapDetectionResult:
        """
        检测小地图中的关键元素

        Args:
            frame_minimap: 小地图 BGR 图像
            confidence_threshold: 置信度阈值（可覆盖默认）

        Returns:
            MinimapDetectionResult
        """
        if frame_minimap is None or frame_minimap.size == 0:
            logger.error("MinimapDetector: frame_minimap为空或无效")
            return MinimapDetectionResult(None, None, [])

        conf = confidence_threshold or self.conf_threshold_

        # 使用结构化检测
        detections = self.engine_.DetectStructured(frame_minimap, confidence_threshold=conf)

        # 位置提取
        self_pos = self._extract_center(detections, CLS_SELF)
        enemy_flag_pos = self._extract_center(detections, CLS_ENEMY_FLAG)

        return MinimapDetectionResult(self_pos, enemy_flag_pos, detections)

    # -----------------------------------------------------------
    @staticmethod
    def _extract_center(detections: List[Detection], cls_id: int) -> Optional[Tuple[float, float]]:
        """从检测列表中筛选某个类别，并取最高置信度的中心点"""
        filtered = [d for d in detections if d.cls == cls_id]
        if not filtered:
            return None

        best = max(filtered, key=lambda d: d.confidence)
        return best.center

    # -----------------------------------------------------------
    def Reset(self):
        pass
