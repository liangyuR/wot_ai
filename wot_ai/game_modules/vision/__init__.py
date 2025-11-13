#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测模块（共享的检测功能）
"""

from .detection_engine import DetectionEngine

from .detection.minimap_anchor_detector import MinimapAnchorDetector
from .detection.minimap_detector import MinimapDetector

__all__ = [
    'DetectionEngine',
    'MinimapAnchorDetector',
    'MinimapDetector',
]

