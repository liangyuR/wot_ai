#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
卡顿检测器 - 基于时间窗口内累计行程判断是否卡死

核心逻辑：
- 记录时间窗口内的位置历史
- 计算累计行程（相邻帧位移之和），而非首尾直线距离
- 累计行程 < 阈值 → 判定卡顿
"""

import math
import time
from typing import Optional, Tuple, List
from collections import deque
from loguru import logger
from src.utils.global_path import GetGlobalConfig


class StuckDetector:
    """卡顿检测器"""

    # 最小检测时间（秒），历史数据不足此时间则不判定
    MIN_DETECTION_TIME = 3.0

    def __init__(self):
        config = GetGlobalConfig()
        stuck_cfg = getattr(config, "stuck_detection", None)

        # 配置参数
        self.time_window_: float = self._getConfigValue(
            stuck_cfg, "time_window_s", 30.0
        )
        self.dist_threshold_: float = self._getConfigValue(
            stuck_cfg, "dist_threshold_px", 0.5
        )

        # 位置历史：[(x, y, timestamp), ...]
        self._history: deque = deque()

        # 状态
        self._is_stuck: bool = False
        self._stuck_count: int = 0
        self._last_travel: float = 0.0  # 上次计算的累计行程（调试用）

    @staticmethod
    def _getConfigValue(cfg, key: str, default):
        """兼容 dict 和 object 两种配置访问方式"""
        if cfg is None:
            return default
        if hasattr(cfg, key):
            val = getattr(cfg, key, None)
            return val if val is not None else default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return default

    def update(self, pos: Tuple[float, float]) -> bool:
        """
        更新位置并返回是否卡顿

        Args:
            pos: 当前坐标 (x, y)

        Returns:
            True 如果检测到卡顿，否则 False
        """
        now = time.perf_counter()
        x, y = pos

        # 添加当前位置
        self._history.append((x, y, now))

        # 清理过期记录（保留时间窗口内的数据）
        cutoff = now - self.time_window_
        while self._history and self._history[0][2] < cutoff:
            self._history.popleft()

        # 数据不足，不判定
        if len(self._history) < 2:
            self._is_stuck = False
            return False

        # 时间跨度不足最小检测时间，不判定
        time_span = now - self._history[0][2]
        if time_span < self.MIN_DETECTION_TIME:
            self._is_stuck = False
            return False

        # 计算累计行程（相邻点之间的距离之和）
        travel = 0.0
        prev = self._history[0]
        for curr in list(self._history)[1:]:
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            travel += math.hypot(dx, dy)
            prev = curr

        self._last_travel = travel

        # 判定：累计行程 < 阈值 → 卡顿
        self._is_stuck = travel < self.dist_threshold_
        return self._is_stuck

    def reset(self) -> None:
        """重置检测状态（路径重规划或脱困后调用）"""
        self._history.clear()
        self._is_stuck = False
        self._last_travel = 0.0

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
        time_span = 0.0
        if len(self._history) >= 2:
            time_span = self._history[-1][2] - self._history[0][2]

        return {
            "is_stuck": self._is_stuck,
            "stuck_count": self._stuck_count,
            "travel": self._last_travel,
            "threshold": self.dist_threshold_,
            "time_span": time_span,
            "time_window": self.time_window_,
            "history_size": len(self._history),
        }

    # 兼容旧接口
    @property
    def is_stuck(self) -> bool:
        return self._is_stuck

    @property
    def is_stuck_(self) -> bool:
        return self._is_stuck

    @property
    def stuck_frames(self) -> int:
        """兼容旧接口，返回估算帧数"""
        if len(self._history) >= 2:
            time_span = self._history[-1][2] - self._history[0][2]
            return int(time_span * 30)
        return 0

    @property
    def stuck_frames_(self) -> int:
        return self.stuck_frames

    @property
    def stuck_count_(self) -> int:
        return self._stuck_count
