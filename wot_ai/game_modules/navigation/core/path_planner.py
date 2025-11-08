#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径规划模块：实现 A* 算法进行路径规划
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import heapq
import numpy as np
# 统一导入机制
from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple
setup_python_path()

SetupLogger = None
logger_module, _ = try_import_multiple([
    'wot_ai.game_modules.common.utils.logger',
    'game_modules.common.utils.logger',
    'common.utils.logger',
    'yolo.utils.logger'
])
if logger_module is not None:
    SetupLogger = getattr(logger_module, 'SetupLogger', None)

if SetupLogger is None:
    from ...common.utils.logger import SetupLogger

logger = SetupLogger(__name__)


class BasePlanner(ABC):
    """路径规划器基类（抽象接口）"""
    
    @abstractmethod
    def Plan(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        规划路径
        
        Args:
            grid: 栅格地图（0=可通行，1=不可通行）
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
        
        Returns:
            路径坐标列表，如果无法到达则返回空列表
        """
        raise NotImplementedError


class AStarPlanner(BasePlanner):
    """A* 算法路径规划器"""
    
    def __init__(self, enable_smoothing: bool = True, smooth_weight: float = 0.3):
        """
        初始化 A* 规划器
        
        Args:
            enable_smoothing: 是否启用路径平滑
            smooth_weight: 平滑权重（0.0-1.0）
        """
        self.enable_smoothing_ = enable_smoothing
        self.smooth_weight_ = smooth_weight
        
        # 四方向移动
        self.directions_ = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    def Plan(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        规划路径（实现基类接口）
        
        Args:
            grid: 栅格地图
            start: 起点
            goal: 终点
        
        Returns:
            路径坐标列表
        """
        path = self.AStar(grid, start, goal)
        
        if self.enable_smoothing_ and len(path) > 2:
            path = self.SmoothPath(path, self.smooth_weight_)
        
        return path
    
    def AStar(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* 算法核心实现
        
        Args:
            grid: 栅格地图
            start: 起点
            goal: 终点
        
        Returns:
            路径坐标列表
        """
        height, width = grid.shape
        
        # 检查边界
        if (start[0] < 0 or start[0] >= width or start[1] < 0 or start[1] >= height or
            goal[0] < 0 or goal[0] >= width or goal[1] < 0 or goal[1] >= height):
            logger.warning("起点或终点超出地图范围")
            return []
        
        # 检查起点和终点是否为障碍物
        if grid[start[1], start[0]] == 1:
            logger.warning("起点位于障碍物上")
            return []
        
        if grid[goal[1], goal[0]] == 1:
            logger.warning("终点位于障碍物上")
            return []
        
        # 如果起点就是终点
        if start == goal:
            return [start]
        
        # 优先队列：(f_score, g_score, current, came_from)
        open_set = [(0, 0, start, None)]
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current_g, current, came_from_pos = heapq.heappop(open_set)
            
            # 到达终点
            if current == goal:
                # 重建路径
                path = [goal]
                pos = current
                while pos in came_from:
                    pos = came_from[pos]
                    path.append(pos)
                path.reverse()
                return path
            
            # 探索邻居
            for dx, dy in self.directions_:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                # 检查边界
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                
                # 检查障碍物
                if grid[ny, nx] == 1:
                    continue
                
                # 计算新的g_score
                new_g = current_g + 1
                
                # 如果找到更短的路径
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    h_score = self.Heuristic(neighbor, goal)
                    f_score = new_g + h_score
                    
                    heapq.heappush(open_set, (f_score, new_g, neighbor, current))
                    came_from[neighbor] = current
        
        # 无法到达终点
        logger.warning("无法找到从起点到终点的路径")
        return []
    
    def Heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        启发式函数（曼哈顿距离）
        
        Args:
            a: 点A坐标
            b: 点B坐标
        
        Returns:
            曼哈顿距离
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def SmoothPath(self, path: List[Tuple[int, int]], weight: float = 0.3) -> List[Tuple[int, int]]:
        """
        路径平滑化：去除锯齿状路径
        
        Args:
            path: 原始路径
            weight: 平滑权重
        
        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_pos = path[i + 1]
            
            # 移动平均平滑
            new_x = curr[0] + weight * (next_pos[0] - prev[0])
            new_y = curr[1] + weight * (next_pos[1] - prev[1])
            
            smoothed.append((int(new_x), int(new_y)))
        
        smoothed.append(path[-1])
        
        return smoothed

