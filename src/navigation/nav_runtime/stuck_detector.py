#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
卡顿检测器 - 基于固定时间间隔和直线距离判断是否卡死

核心逻辑：
- 记录检测起点坐标和时间戳
- 固定时间间隔后，计算当前位置与起点的直线距离
- 直线距离 < 阈值 → 判定卡顿
"""

import math
import time
from typing import Tuple, Optional
from loguru import logger
from src.utils.global_path import GetGlobalConfig


class StuckDetector:
    """卡顿检测器"""

    def __init__(self):
        stuck_cfg = GetGlobalConfig().stuck_detection

        # 配置参数
        self.check_interval_: float = stuck_cfg.check_interval_s
        self.dist_threshold_: float = stuck_cfg.dist_threshold_px

        # 检测起点
        self._checkpoint_pos: Optional[Tuple[float, float]] = None
        self._checkpoint_time: float = 0.0

        # 状态
        self._is_stuck: bool = False
        self._stuck_count: int = 0
        self._last_distance: float = 0.0  # 上次计算的直线距离（调试用）
        
    def update(self, pos: Tuple[float, float]) -> bool:
        """
        更新位置并返回是否卡顿

        Args:
            pos: 当前坐标 (x, y)

        Returns:
            True 如果检测到卡顿，否则 False
        """
        now = time.perf_counter()

        # 首次调用，设置检测起点
        if self._checkpoint_pos is None:
            self._checkpoint_pos = pos
            self._checkpoint_time = now
            self._is_stuck = False
            return False

        # 未到检测时间，不判定
        elapsed = now - self._checkpoint_time
        if elapsed < self.check_interval_:
            self._is_stuck = False
            return False

        # 计算直线距离
        dx = pos[0] - self._checkpoint_pos[0]
        dy = pos[1] - self._checkpoint_pos[1]
        distance = math.hypot(dx, dy)
        self._last_distance = distance

        # 重置检测起点（无论是否卡顿）
        self._checkpoint_pos = pos
        self._checkpoint_time = now

        # 判定：距离 < 阈值 → 卡顿
        self._is_stuck = distance < self.dist_threshold_
        if self._is_stuck:
            logger.debug(
                f"卡顿检测: 间隔={elapsed:.1f}s, 移动距离={distance:.1f}px < 阈值={self.dist_threshold_}px"
            )
        return self._is_stuck

    def reset(self) -> None:
        """重置检测状态（路径重规划或脱困后调用）"""
        self._checkpoint_pos = None
        self._checkpoint_time = 0.0
        self._is_stuck = False
        self._last_distance = 0.0

    def incrementStuckCount(self) -> None:
        """增加连续卡顿计数（由上层在处理卡顿后调用）"""
        self._stuck_count += 1
        logger.debug(f"卡顿计数: {self._stuck_count}")

    def resetStuckCount(self) -> None:
        """重置连续卡顿计数（成功移动后调用）"""
        if self._stuck_count > 0:
            logger.debug(f"重置卡顿计数（之前: {self._stuck_count}）")
        self._stuck_count = 0

    def getStuckCount(self) -> int:
        """获取连续卡顿次数"""
        return self._stuck_count

    def getDebugInfo(self) -> dict:
        """获取调试信息"""
        elapsed = 0.0
        if self._checkpoint_pos is not None:
            elapsed = time.perf_counter() - self._checkpoint_time

        return {
            "is_stuck": self._is_stuck,
            "stuck_count": self._stuck_count,
            "distance": self._last_distance,
            "threshold": self.dist_threshold_,
            "elapsed": elapsed,
            "check_interval": self.check_interval_,
            "checkpoint": self._checkpoint_pos,
        }

    @property
    def stuck_count_(self) -> int:
        return self._stuck_count
