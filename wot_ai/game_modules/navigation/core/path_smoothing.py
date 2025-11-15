#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径平滑模块：Line-of-sight平滑

功能：
- 使用Bresenham算法检测两点间是否无障碍
- 使用LOS简化路径，去除多余折线
"""

import numpy as np
from typing import List, Tuple
from loguru import logger

Coord = Tuple[int, int]  # (x, y)


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

