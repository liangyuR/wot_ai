#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计收集器：负责统计信息收集和FPS计算
"""

# 标准库导入
from typing import Dict, Any
import time

# 本地模块导入
from ..common.imports import GetLogger

logger = GetLogger()(__name__)


class StatsCollector:
    """统计收集器：收集和计算导航系统的统计信息"""
    
    def __init__(self):
        """初始化统计收集器"""
        self.stats_: Dict[str, Any] = {
            'frame_count': 0,
            'detection_count': 0,
            'path_planning_count': 0,
            'navigation_count': 0,
            'fps': 0.0
        }
        
        # FPS计算相关
        self.fps_interval_ = 1.0
        self.fps_count_ = 0
        self.fps_start_time_ = time.time()
    
    def UpdateFrameCount(self) -> None:
        """更新帧计数"""
        self.stats_['frame_count'] += 1
    
    def UpdateDetectionCount(self) -> None:
        """更新检测计数"""
        self.stats_['detection_count'] += 1
    
    def UpdatePathPlanningCount(self) -> None:
        """更新路径规划计数"""
        self.stats_['path_planning_count'] += 1
    
    def UpdateNavigationCount(self) -> None:
        """更新导航计数"""
        self.stats_['navigation_count'] += 1
    
    def UpdateFps(self) -> bool:
        """
        更新FPS（如果达到更新间隔）
        
        Returns:
            是否更新了FPS
        """
        self.fps_count_ += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time_ >= self.fps_interval_:
            self.stats_['fps'] = self.fps_count_ / (current_time - self.fps_start_time_)
            self.fps_count_ = 0
            self.fps_start_time_ = current_time
            return True
        
        return False
    
    def GetStats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats_.copy()
    
    def Reset(self) -> None:
        """重置所有统计信息"""
        self.stats_ = {
            'frame_count': 0,
            'detection_count': 0,
            'path_planning_count': 0,
            'navigation_count': 0,
            'fps': 0.0
        }
        self.fps_count_ = 0
        self.fps_start_time_ = time.time()

