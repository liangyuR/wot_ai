#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional, Tuple, Deque
from collections import deque
from src.utils.global_path import GetGlobalConfig


class StuckDetector:
    """
    用于检测坦克是否卡顿（基于时间窗口内的总位移）
    
    改进：使用时间窗口检测，避免低速移动时的误判
    - 记录参考位置（N帧前的位置）
    - 计算从参考位置到当前位置的总位移
    - 如果总位移小于阈值，才判定为卡死
    """

    def __init__(self):
        config = GetGlobalConfig()
        ctrl_cfg = config.control
        
        # 改为固定参数：6秒内位移小于10px视为卡死
        self.time_window = 6.0  # 秒
        self.dist_threshold = 10.0  # 像素

        # 记录历史位置（用于时间窗口检测）
        # 存储 (pos, timestamp) 元组
        # maxlen 不再严格限制，而是通过时间清理
        self._pos_history: Deque[Tuple[Tuple[float, float], float]] = deque()
        
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

        current_time = time.perf_counter()
        
        if self.last_pos is None:
            self.last_pos = pos
            self.stuck_frames = 0
            self.is_stuck = False
            self._pos_history.append((pos, current_time))
            return False

        # 添加当前位置到历史记录
        self._pos_history.append((pos, current_time))
        
        # 清理过期记录（超过时间窗口的）
        while self._pos_history and (current_time - self._pos_history[0][1] > self.time_window):
            self._pos_history.popleft()
        
        # 如果历史记录时间跨度不足窗口时间的 80%，暂不判定（给一些启动缓冲）
        if not self._pos_history:
             # Should not happen as we just appended
            return False
            
        start_time = self._pos_history[0][1]
        time_span = current_time - start_time
        
        if time_span < self.time_window * 0.8:
            self.last_pos = pos
            self.stuck_frames = 0
            self.is_stuck = False
            return False

        # 计算时间窗口内的总位移
        # 参考位置：历史记录中最早的点
        ref_pos, _ = self._pos_history[0]
        dx = pos[0] - ref_pos[0]
        dy = pos[1] - ref_pos[1]
        total_dist = math.hypot(dx, dy)
        
        # 改进的卡死判定逻辑：
        # 如果在时间窗口（6秒）内，总位移小于阈值（10px），则判定为卡死
        if total_dist < self.dist_threshold:
            self.is_stuck = True
            # 这里的 stuck_frames 仅用于调试显示，保持兼容
            self.stuck_frames = int(time_span * 30) 
        else:
            # 有显著移动（总位移达到阈值），重置状态
            self.is_stuck = False
            self.stuck_frames = 0

        self.last_pos = pos

        return self.is_stuck

    def reset(self):
        """重新开始卡顿检测（路径重规划时调用）"""
        self.stuck_frames = 0
        self.is_stuck = False
        self.last_pos = None
        self._pos_history.clear()
