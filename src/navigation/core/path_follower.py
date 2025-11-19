#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径跟随模块

封装路径跟随逻辑，提供路径上最近点查找和目标点选择功能。
"""

import math
from typing import List, Tuple, Optional


class PathFollower:
    """路径跟随器"""
    
    def find_nearest_point(
        self,
        current_pos: Tuple[float, float],
        path_world: List[Tuple[float, float]],
        start_idx: int = 0
    ) -> Tuple[int, Tuple[float, float], float]:
        """
        找到路径上距离当前位置最近的点（路径跟随逻辑）
        
        Args:
            current_pos: 当前位置 (x, y)（世界坐标）
            path_world: 路径点列表 [(x, y), ...]（世界坐标）
            start_idx: 搜索起始索引，避免回溯
        
        Returns:
            (nearest_idx, nearest_point, distance): 最近点的索引、坐标和距离
        """
        if not path_world or len(path_world) == 0:
            return (0, current_pos, float('inf'))
        
        min_dist = float('inf')
        nearest_idx = start_idx
        nearest_point = path_world[start_idx] if start_idx < len(path_world) else path_world[0]
        
        # 从start_idx开始搜索，避免回溯
        for i in range(start_idx, len(path_world)):
            point = path_world[i]
            dist = math.sqrt(
                (point[0] - current_pos[0]) ** 2 + 
                (point[1] - current_pos[1]) ** 2
            )
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                nearest_point = point
        
        # 也检查线段上的最近点（更精确）
        for i in range(start_idx, len(path_world) - 1):
            p1 = path_world[i]
            p2 = path_world[i + 1]
            
            # 计算到线段的最近点
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            line_len_sq = dx * dx + dy * dy
            
            if line_len_sq < 1e-6:
                continue
            
            # 计算投影参数t
            t = ((current_pos[0] - p1[0]) * dx + (current_pos[1] - p1[1]) * dy) / line_len_sq
            t = max(0.0, min(1.0, t))  # 限制在线段内
            
            # 投影点
            proj_x = p1[0] + t * dx
            proj_y = p1[1] + t * dy
            proj_point = (proj_x, proj_y)
            
            # 计算距离
            dist = math.sqrt(
                (proj_point[0] - current_pos[0]) ** 2 + 
                (proj_point[1] - current_pos[1]) ** 2
            )
            
            if dist < min_dist:
                min_dist = dist
                # 如果投影点更接近p2，使用下一个索引
                if t > 0.5:
                    nearest_idx = min(i + 1, len(path_world) - 1)
                else:
                    nearest_idx = i
                nearest_point = proj_point
        
        return (nearest_idx, nearest_point, min_dist)
    
    def get_target_point(
        self,
        current_pos: Tuple[float, float],
        path_world: List[Tuple[float, float]],
        current_idx: int,
        offset: int = 20
    ) -> Optional[Tuple[float, float]]:
        """
        获取前瞻目标点
        
        Args:
            current_pos: 当前位置 (x, y)
            path_world: 路径点列表
            current_idx: 当前路径索引
            offset: 前瞻偏移量（路径点数量）
        
        Returns:
            目标点坐标，如果路径已结束则返回None
        """
        if not path_world or len(path_world) == 0:
            return None
        
        target_idx = min(current_idx + offset, len(path_world) - 1)
        if target_idx < len(path_world):
            return path_world[target_idx]
        
        return None

