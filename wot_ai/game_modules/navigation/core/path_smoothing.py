#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径平滑模块：Line-of-sight平滑和Catmull-Rom曲线平滑

功能：
- 使用Bresenham算法检测两点间是否无障碍
- 使用LOS简化路径，去除多余折线
- Catmull-Rom曲线平滑：生成连续、可微分的曲线轨迹
"""

import numpy as np
import math
from typing import List, Tuple
from loguru import logger

from .path_simplify import simplify_path_direction, rdp_simplify
from .catmull_rom import catmull_rom_interpolate

Coord = Tuple[int, int]  # (x, y)
Point = Tuple[float, float]  # (x, y)


def line_of_sight(occ_grid: np.ndarray, p1: Coord, p2: Coord) -> bool:
    """
    简单 Bresenham 视线检测:
        occ_grid[y, x] == 1 表示障碍

    Args:
        occ_grid: HxW 二值障碍图，1=障碍，0=可通行
        p1: 起点坐标 (x, y)
        p2: 终点坐标 (x, y)

    Returns:
        True: 两点之间直线无障碍
    """
    x0, y0 = p1
    x1, y1 = p2

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    x, y = x0, y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1

    error = dx - dy
    dx *= 2
    dy *= 2

    h, w = occ_grid.shape

    for _ in range(n):
        # 检查边界
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        
        # 检查障碍
        if occ_grid[y, x] != 0:
            return False

        if (x, y) == (x1, y1):
            break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return True


def smooth_path_los(path: List[Coord], occ_grid: np.ndarray) -> List[Coord]:
    """
    基于 line-of-sight 的路径简化：
        尽量用更远的点替代中间折线点。

    Args:
        path: 原始路径 [(x, y), ...]
        occ_grid: HxW 二值障碍图，1=障碍，0=可通行（建议用膨胀后的障碍图）

    Returns:
        简化后的路径
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0
    n = len(path)

    while i < n - 1:
        j = n - 1
        # 从尾部往回找最远可直连点
        while j > i + 1 and not line_of_sight(occ_grid, path[i], path[j]):
            j -= 1
        smoothed.append(path[j])
        i = j

    logger.debug(f"LOS平滑: 原始路径长度={len(path)}, 平滑后={len(smoothed)}")
    return smoothed


def compute_heading_and_curvature(path: List[Point]) -> Tuple[List[float], List[float]]:
    """
    计算路径上每个点的heading（朝向）和curvature（曲率）
    
    Args:
        path: 路径点列表 [(x, y), ...]
    
    Returns:
        (headings, curvatures): heading列表（度），curvature列表（度）
    """
    if len(path) < 2:
        return ([], [])
    
    headings = []
    curvatures = []
    
    # 计算heading（每个点的朝向）
    for i in range(len(path)):
        if i == 0:
            # 第一个点：使用到下一个点的方向
            if len(path) > 1:
                dx = path[1][0] - path[0][0]
                dy = path[1][1] - path[0][1]
                heading = math.degrees(math.atan2(dy, dx))
            else:
                heading = 0.0
        elif i == len(path) - 1:
            # 最后一个点：使用从前一个点的方向
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            heading = math.degrees(math.atan2(dy, dx))
        else:
            # 中间点：使用前后两点的平均方向
            dx1 = path[i][0] - path[i - 1][0]
            dy1 = path[i][1] - path[i - 1][1]
            dx2 = path[i + 1][0] - path[i][0]
            dy2 = path[i + 1][1] - path[i][1]
            
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            
            # 计算平均角度（处理跨越-π/π边界的情况）
            avg_angle = (angle1 + angle2) / 2.0
            if abs(angle2 - angle1) > math.pi:
                avg_angle += math.pi
            
            heading = math.degrees(avg_angle)
        
        headings.append(heading)
    
    # 计算curvature（曲率，即相邻heading的变化）
    for i in range(len(path)):
        if i == 0 or i == len(path) - 1:
            curvatures.append(0.0)
        else:
            # 计算heading变化
            h1 = headings[i - 1]
            h2 = headings[i]
            h3 = headings[i + 1]
            
            # 计算两段方向变化
            diff1 = h2 - h1
            if diff1 > 180:
                diff1 -= 360
            elif diff1 < -180:
                diff1 += 360
            
            diff2 = h3 - h2
            if diff2 > 180:
                diff2 -= 360
            elif diff2 < -180:
                diff2 += 360
            
            # 曲率 = 方向变化率
            curvature = abs(diff1) + abs(diff2)
            curvatures.append(curvature)
    
    return (headings, curvatures)


def check_and_refine_curvature(
    path: List[Point],
    curvature_threshold_deg: float = 40.0,
    num_points_per_segment: int = 15
) -> List[Point]:
    """
    检查路径曲率，对急弯段重新插值
    
    Args:
        path: 原始路径
        curvature_threshold_deg: 曲率阈值（度），超过此值认为是急弯
        num_points_per_segment: 每段重新插值的点数
    
    Returns:
        优化后的路径
    """
    if len(path) < 3:
        return path
    
    _, curvatures = compute_heading_and_curvature(path)
    
    # 找出急弯段
    sharp_turns = []
    for i in range(1, len(path) - 1):
        if curvatures[i] > curvature_threshold_deg:
            sharp_turns.append(i)
    
    if not sharp_turns:
        # 没有急弯，直接返回
        return path
    
    logger.debug(f"检测到{len(sharp_turns)}个急弯点，曲率阈值={curvature_threshold_deg}°")
    
    # 对急弯段进行重新插值
    result = []
    last_refined_idx = -1
    
    for turn_idx in sharp_turns:
        # 添加急弯点之前的所有点
        if turn_idx > last_refined_idx + 1:
            result.extend(path[last_refined_idx + 1:turn_idx])
        
        # 对急弯段进行局部插值（使用前后各2个点）
        start_idx = max(0, turn_idx - 2)
        end_idx = min(len(path), turn_idx + 3)
        segment = path[start_idx:end_idx]
        
        # 使用Catmull-Rom插值
        if len(segment) >= 2:
            interpolated = catmull_rom_interpolate(segment, num_points_per_segment)
            # 跳过第一个点（已经在result中）
            if result and interpolated:
                result.extend(interpolated[1:])
            else:
                result.extend(interpolated)
        
        last_refined_idx = end_idx - 1
    
    # 添加剩余的点
    if last_refined_idx < len(path) - 1:
        result.extend(path[last_refined_idx + 1:])
    
    return result


def smooth_path(
    path: List[Point],
    simplify_method: str = "direction",
    simplify_threshold: float = 2.0,
    num_points_per_segment: int = 15,
    curvature_threshold_deg: float = 40.0,
    check_curvature: bool = True
) -> List[Point]:
    """
    主入口函数：对A*路径进行完整平滑处理
    
    流程：
    1. 路径简化（去噪/点简化）
    2. Catmull-Rom插值
    3. 曲率检查（可选）
    
    Args:
        path: 原始A*路径 [(x, y), ...]
        simplify_method: 简化方法，"direction"或"rdp"
        simplify_threshold: 简化阈值
            - direction方法：角度阈值（度）
            - rdp方法：距离阈值（像素）
        num_points_per_segment: 每段曲线采样点数，默认15
        curvature_threshold_deg: 曲率阈值（度），超过此值认为是急弯
        check_curvature: 是否进行曲率检查
    
    Returns:
        平滑后的路径
    """
    if len(path) < 2:
        return path
    
    logger.debug(f"开始路径平滑: 原始长度={len(path)}")
    
    # 步骤1：路径简化
    if simplify_method == "direction":
        simplified = simplify_path_direction(path, simplify_threshold)
    elif simplify_method == "rdp":
        simplified = rdp_simplify(path, simplify_threshold)
    else:
        logger.warning(f"未知的简化方法: {simplify_method}，跳过简化")
        simplified = path
    
    logger.debug(f"路径简化完成: {len(path)} -> {len(simplified)}")
    
    # 步骤2：Catmull-Rom插值
    if len(simplified) >= 2:
        interpolated = catmull_rom_interpolate(simplified, num_points_per_segment)
    else:
        interpolated = simplified
    
    logger.debug(f"Catmull-Rom插值完成: {len(simplified)} -> {len(interpolated)}")
    
    # 步骤3：曲率检查（可选）
    if check_curvature and len(interpolated) >= 3:
        final_path = check_and_refine_curvature(
            interpolated,
            curvature_threshold_deg,
            num_points_per_segment
        )
        logger.debug(f"曲率检查完成: {len(interpolated)} -> {len(final_path)}")
    else:
        final_path = interpolated
    
    logger.info(f"路径平滑完成: {len(path)} -> {len(final_path)}")
    return final_path

