#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple


class StuckDetector:
    """
    用于检测坦克是否卡顿（连续多个 tick 移动距离不足）
    """

    def __init__(self, move_threshold: float, frame_threshold: int):
        """
        Args:
            move_threshold: 移动距离阈值（px），小于此值视为“这一帧未移动”
            frame_threshold: 连续多少帧未移动视为卡顿
        """
        self.move_th = move_threshold
        self.frame_th = frame_threshold

        self.last_pos: Optional[Tuple[float, float]] = None
        self.stuck_frames: int = 0
        self.is_stuck: bool = False

    def update(self, pos: Tuple[float, float]) -> bool:
        """
        更新位置并返回是否卡顿

        Args:
            pos: 当前坐标 (x, y)
        
        Returns:
            是否卡顿（True/False）
        """

        if self.last_pos is None:
            self.last_pos = pos
            self.stuck_frames = 0
            self.is_stuck = False
            return False

        dx = pos[0] - self.last_pos[0]
        dy = pos[1] - self.last_pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.move_th:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        self.last_pos = pos
        self.is_stuck = (self.stuck_frames >= self.frame_th)

        return self.is_stuck

    def reset(self):
        """重新开始卡顿检测（路径重规划时调用）"""
        self.stuck_frames = 0
        self.is_stuck = False
        self.last_pos = None
