#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带代价图的A*路径规划器

功能：
- 基于cost_map的A*算法
- 支持8邻接移动
- 使用欧氏距离启发函数
"""

import heapq
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger

Coord = Tuple[int, int]  # (x, y)


def astar_with_cost(
    cost_map: np.ndarray,
    start: Coord,
    goal: Coord,
) -> List[Coord]:
    """
    在 cost_map 上做 A*:
        - cost_map[y, x] 为移动代价，障碍为 np.inf

    Args:
        cost_map: HxW float32数组，移动代价（障碍为np.inf）
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)

    Returns:
        path: [(x, y), ...]，从 start 到 goal，找不到则 []
    """
    h, w = cost_map.shape
    sx, sy = start
    gx, gy = goal

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h

    def is_blocked(x: int, y: int) -> bool:
        return not np.isfinite(cost_map[y, x])

    def heuristic(x: int, y: int) -> float:
        # 欧氏距离
        return ((x - gx)**2 + (y - gy)**2) ** 0.5

    # 8 邻接
    neighbors = [
        (-1,  0, 1.0),   # 左
        ( 1,  0, 1.0),   # 右
        ( 0, -1, 1.0),   # 上
        ( 0,  1, 1.0),   # 下
        (-1, -1, 2**0.5),  # 左上
        (-1,  1, 2**0.5),  # 左下
        ( 1, -1, 2**0.5),  # 右上
        ( 1,  1, 2**0.5),  # 右下
    ]

    g_score = np.full_like(cost_map, np.inf, dtype=np.float32)
    g_score[sy, sx] = 0.0

    came_from: dict[Coord, Coord] = {}

    open_heap = []
    heapq.heappush(open_heap, (heuristic(sx, sy), (sx, sy)))

    visited = np.zeros_like(cost_map, dtype=bool)

    nodes_explored = 0

    while open_heap:
        _, (x, y) = heapq.heappop(open_heap)

        if visited[y, x]:
            continue
        visited[y, x] = True
        nodes_explored += 1

        if (x, y) == (gx, gy):
            # 回溯路径
            path: List[Coord] = []
            cur = (gx, gy)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append((sx, sy))
            path.reverse()
            logger.debug(f"A*规划成功: 路径长度={len(path)}, 探索节点数={nodes_explored}")
            return path

        for dx, dy, step_cost in neighbors:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny) or is_blocked(nx, ny):
                continue

            # 移动代价 = 步长 * cost_map
            move_cost = step_cost * float(cost_map[ny, nx])
            tentative_g = g_score[y, x] + move_cost
            if tentative_g < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g
                came_from[(nx, ny)] = (x, y)
                f = tentative_g + heuristic(nx, ny)
                heapq.heappush(open_heap, (f, (nx, ny)))

    # 无路径
    logger.warning(f"A*规划失败: 无法找到从{start}到{goal}的路径, 探索节点数={nodes_explored}")
    return []

