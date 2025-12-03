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
        
        move_threshold = ctrl_cfg.stuck_threshold
        frame_threshold = ctrl_cfg.stuck_frames_threshold
        self.move_th = move_threshold
        self.frame_th = frame_threshold

        # 记录历史位置（用于时间窗口检测）
        # 存储 (pos, timestamp) 元组
        self._pos_history: Deque[Tuple[Tuple[float, float], float]] = deque(maxlen=frame_threshold)
        
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
        
        # 如果历史记录不足，无法判断
        if len(self._pos_history) < self.frame_th:
            self.last_pos = pos
            self.stuck_frames = 0
            self.is_stuck = False
            return False

        # 计算时间窗口内的总位移
        # 参考位置：N帧前的位置（历史记录的第一个）
        ref_pos, ref_time = self._pos_history[0]
        dx = pos[0] - ref_pos[0]
        dy = pos[1] - ref_pos[1]
        total_dist = math.hypot(dx, dy)
        
        # 计算单帧位移（用于快速检测）
        dx_frame = pos[0] - self.last_pos[0]
        dy_frame = pos[1] - self.last_pos[1]
        frame_dist = math.hypot(dx_frame, dy_frame)
        
        # 改进的卡死判定逻辑：
        # 使用时间窗口检测，避免低速移动时的误判
        # 总位移阈值：允许最低速度移动（每帧至少0.3像素）
        # 如果坦克以最低速度移动（每帧0.3像素），200帧后应该移动60像素
        # 所以总位移阈值设置为：frame_threshold * 0.3
        total_threshold = self.frame_th * 0.3
        
        # 如果总位移很小（说明在时间窗口内几乎没有移动）
        # 直接判定为卡死（不需要累积，因为已经在N帧窗口内检测）
        if total_dist < total_threshold:
            self.is_stuck = True
            self.stuck_frames = self.frame_th  # 标记为已卡死
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
