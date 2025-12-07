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
    - 支持连续卡顿计数，供上层实现升级脱困策略
    """

    def __init__(self):
        config = GetGlobalConfig()
        stuck_cfg = getattr(config, "stuck_detection", None)
        
        # 从配置读取参数，提供默认值
        if stuck_cfg is not None:
            self.time_window_ = getattr(stuck_cfg, "time_window_s", 7.0)
            self.dist_threshold_ = getattr(stuck_cfg, "dist_threshold_px", 10.0)
        else:
            # 默认参数：7秒内位移小于10px视为卡死
            self.time_window_ = 7.0
            self.dist_threshold_ = 10.0

        # 记录历史位置（用于时间窗口检测）
        # 存储 (pos, timestamp) 元组
        # maxlen 不再严格限制，而是通过时间清理
        self._pos_history_: Deque[Tuple[Tuple[float, float], float]] = deque()
        
        self.last_pos_: Optional[Tuple[float, float]] = None
        self.stuck_frames_: int = 0
        self.is_stuck_: bool = False
        
        # 连续卡顿计数（每次检测到卡顿并处理后由上层调用 incrementStuckCount）
        self.stuck_count_: int = 0

    def update(self, pos: Tuple[float, float]) -> bool:
        """
        更新位置并返回是否卡顿

        Args:
            pos: 当前坐标 (x, y)
        
        Returns:
            是否卡顿（True/False）
        """
        current_time = time.perf_counter()
        
        if self.last_pos_ is None:
            self.last_pos_ = pos
            self.stuck_frames_ = 0
            self.is_stuck_ = False
            self._pos_history_.append((pos, current_time))
            return False

        # 添加当前位置到历史记录
        self._pos_history_.append((pos, current_time))
        
        # 清理过期记录（超过时间窗口的）
        while self._pos_history_ and (current_time - self._pos_history_[0][1] > self.time_window_):
            self._pos_history_.popleft()
        
        # 如果历史记录时间跨度不足窗口时间的 80%，暂不判定（给一些启动缓冲）
        if not self._pos_history_:
            # Should not happen as we just appended
            return False
            
        start_time = self._pos_history_[0][1]
        time_span = current_time - start_time
        
        if time_span < self.time_window_ * 0.8:
            self.last_pos_ = pos
            self.stuck_frames_ = 0
            self.is_stuck_ = False
            return False

        # 计算时间窗口内的总位移
        # 参考位置：历史记录中最早的点
        ref_pos, _ = self._pos_history_[0]
        dx = pos[0] - ref_pos[0]
        dy = pos[1] - ref_pos[1]
        total_dist = math.hypot(dx, dy)
        
        # 卡死判定逻辑：
        # 如果在时间窗口内，总位移小于阈值，则判定为卡死
        if total_dist < self.dist_threshold_:
            self.is_stuck_ = True
            # 这里的 stuck_frames 仅用于调试显示，保持兼容
            self.stuck_frames_ = int(time_span * 30)
        else:
            # 有显著移动（总位移达到阈值），重置状态
            self.is_stuck_ = False
            self.stuck_frames_ = 0

        self.last_pos_ = pos

        return self.is_stuck_

    def reset(self) -> None:
        """重新开始卡顿检测（路径重规划时调用）"""
        self.stuck_frames_ = 0
        self.is_stuck_ = False
        self.last_pos_ = None
        self._pos_history_.clear()

    def incrementStuckCount(self) -> None:
        """增加连续卡顿计数（由上层在处理卡顿后调用）"""
        self.stuck_count_ += 1

    def resetStuckCount(self) -> None:
        """重置连续卡顿计数（成功移动一段距离后调用）"""
        self.stuck_count_ = 0

    def getStuckCount(self) -> int:
        """获取连续卡顿次数"""
        return self.stuck_count_

    # 保持兼容性的属性访问
    @property
    def is_stuck(self) -> bool:
        return self.is_stuck_

    @property
    def stuck_frames(self) -> int:
        return self.stuck_frames_
