#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置平滑器：用于稳定检测结果，减少帧间跳动
"""

from collections import deque
from typing import Optional, Tuple
from path_planning.utils.logger import SetupLogger

logger = SetupLogger(__name__)


class PositionSmoother:
    """位置平滑器，使用移动平均滤波"""
    
    def __init__(self, window: int = 5):
        """
        初始化位置平滑器
        
        Args:
            window: 移动平均窗口大小
        """
        self.window_ = window
        self.positions_ = deque(maxlen=window)
    
    def Smooth(self, pos: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        平滑位置
        
        Args:
            pos: 当前位置 (x, y)，可为 None
        
        Returns:
            平滑后的位置，如果输入为 None 则返回 None
        """
        if pos is None:
            return None
        
        self.positions_.append(pos)
        
        if len(self.positions_) == 0:
            return None
        
        x = sum(p[0] for p in self.positions_) / len(self.positions_)
        y = sum(p[1] for p in self.positions_) / len(self.positions_)
        
        return (x, y)
    
    def Reset(self):
        """重置平滑器"""
        self.positions_.clear()
    
    def GetLastPosition(self) -> Optional[Tuple[float, float]]:
        """
        获取最后一个位置
        
        Returns:
            最后一个位置，如果没有则返回 None
        """
        if len(self.positions_) == 0:
            return None
        return self.positions_[-1]

