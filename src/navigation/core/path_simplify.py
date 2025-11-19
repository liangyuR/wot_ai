#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径简化模块：去除A*路径中的多余点

功能：
- 方向一致性判断：合并方向变化小的点
- 可选：RDP算法简化路径
"""

import numpy as np
import math
from typing import List, Tuple
from loguru import logger

Point = Tuple[float, float]  # (x, y)


def simplify_path_direction(path: List[Point], angle_threshold_deg: float = 2.0) -> List[Point]:
    """
    基于方向一致性简化路径
    
    如果连续三点的方向变化 < 阈值，合并中间点
    
    Args:
        path: 原始路径 [(x, y), ...]
        angle_threshold_deg: 角度变化阈值（度），默认2°
    
    Returns:
        简化后的路径
    """
    if len(path) <= 2:
        return path
    
    angle_threshold_rad = math.radians(angle_threshold_deg)
    simplified = [path[0]]
    
    i = 0
    while i < len(path) - 1:
        # 如果只剩最后一个点，直接添加
        if i == len(path) - 2:
            simplified.append(path[-1])
            break
        
        # 计算当前点到下一个点的方向
        p0 = path[i]
        p1 = path[i + 1]
        dx1 = p1[0] - p0[0]
        dy1 = p1[1] - p0[1]
        angle1 = math.atan2(dy1, dx1)
        
        # 检查后续点，找到方向变化超过阈值的点
        j = i + 2
        while j < len(path):
            p2 = path[j]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            if dx2 == 0 and dy2 == 0:
                j += 1
                continue
            
            angle2 = math.atan2(dy2, dx2)
            
            # 计算角度差（处理跨越-π/π边界的情况）
            angle_diff = angle2 - angle1
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            if abs(angle_diff) >= angle_threshold_rad:
                # 方向变化超过阈值，保留p1点
                simplified.append(p1)
                i = j - 1
                break
            
            j += 1
        
        # 如果遍历到末尾都没有找到超过阈值的点，添加最后一个点
        if j >= len(path):
            simplified.append(path[-1])
            break
    
    logger.debug(f"路径简化: 原始长度={len(path)}, 简化后={len(simplified)}")
    return simplified


def rdp_simplify(path: List[Point], epsilon: float) -> List[Point]:
    """
    Ramer-Douglas-Peucker算法简化路径
    
    Args:
        path: 原始路径 [(x, y), ...]
        epsilon: 距离阈值
    
    Returns:
        简化后的路径
    """
    if len(path) <= 2:
        return path
    
    def point_to_line_distance(p: Point, line_start: Point, line_end: Point) -> float:
        """计算点到线段的距离"""
        x0, y0 = p
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 线段长度
        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_len_sq == 0:
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        
        # 计算投影参数t
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_len_sq))
        
        # 投影点
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # 距离
        return math.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)
    
    def rdp_recursive(points: List[Point], start_idx: int, end_idx: int) -> List[int]:
        """递归执行RDP算法，返回保留点的索引"""
        if end_idx - start_idx <= 1:
            return [start_idx, end_idx]
        
        # 找到距离起点和终点连线最远的点
        max_dist = 0
        max_idx = start_idx + 1
        
        for i in range(start_idx + 1, end_idx):
            dist = point_to_line_distance(points[i], points[start_idx], points[end_idx])
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # 如果最大距离小于阈值，只保留起点和终点
        if max_dist < epsilon:
            return [start_idx, end_idx]
        
        # 递归处理左右两段
        left = rdp_recursive(points, start_idx, max_idx)
        right = rdp_recursive(points, max_idx, end_idx)
        
        # 合并结果（去除重复的max_idx）
        result = left[:-1] + right
        return result
    
    # 执行RDP算法
    indices = rdp_recursive(path, 0, len(path) - 1)
    simplified = [path[i] for i in indices]
    
    logger.debug(f"RDP简化: 原始长度={len(path)}, epsilon={epsilon}, 简化后={len(simplified)}")
    return simplified

