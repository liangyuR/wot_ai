#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径规划模块：实现 A* 算法进行路径规划
"""

# 标准库导入
from typing import List, Tuple, Optional, Callable
from collections import deque
import heapq

# 第三方库导入
import numpy as np
from loguru import logger

class AStarPlanner():
    """
    A* 算法路径规划器
    
    实现IPlanner接口，使用A*算法在栅格地图上进行路径规划。
    支持路径平滑化，可配置平滑权重。
    
    示例:
        ```python
        planner = AStarPlanner(enable_smoothing=True, smooth_weight=0.3)
        path = planner.Plan(grid, start=(10, 10), goal=(50, 50))
        ```
    """
    
    def __init__(self, enable_smoothing: bool = True, smooth_weight: float = 0.3):
        """
        初始化 A* 规划器
        
        Args:
            enable_smoothing: 是否启用路径平滑
            smooth_weight: 平滑权重（0.0-1.0）
        
        Raises:
            ValueError: 输入参数无效
        """
        if not isinstance(enable_smoothing, bool):
            raise ValueError("enable_smoothing必须是布尔值")
        if not isinstance(smooth_weight, (int, float)) or smooth_weight < 0 or smooth_weight > 1:
            raise ValueError("smooth_weight必须在0-1之间")
        
        self.enable_smoothing_ = enable_smoothing
        self.smooth_weight_ = smooth_weight
        
        # 四方向移动
        self.directions_ = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    def Plan(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        规划路径（实现IPlanner接口）
        
        Args:
            grid: 栅格地图
            start: 起点
            goal: 终点
        
        Returns:
            路径坐标列表
        
        Raises:
            ValueError: 输入参数无效
        """
        if grid is None or grid.size == 0:
            raise ValueError("grid不能为空")
        if not isinstance(start, tuple) or len(start) != 2:
            raise ValueError("start必须是包含两个整数的元组")
        if not isinstance(goal, tuple) or len(goal) != 2:
            raise ValueError("goal必须是包含两个整数的元组")
        
        try:
            path = self.AStar(grid, start, goal)
            
            if self.enable_smoothing_ and len(path) > 2:
                path = self.SmoothPath(path, self.smooth_weight_)
            
            return path
        except Exception as e:
            error_msg = f"路径规划异常: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _FindNearestWalkable(self, grid: np.ndarray, pos: Tuple[int, int], max_radius: int = 10) -> Optional[Tuple[int, int]]:
        """
        使用 BFS 找到最近的可通行区域（向后兼容包装器）
        
        Args:
            grid: 栅格地图（0=可通行，1=障碍）
            pos: 当前位置 (x, y)
            max_radius: 最大搜索半径
        
        Returns:
            最近的可通行位置，如果找不到则返回 None
        """
        def is_free_obstacle(v: float) -> bool:
            return v == 0
        
        return _find_nearest_free(grid, pos, max_radius, is_free_obstacle)
    
    def AStar(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* 算法核心实现
        
        Args:
            grid: 栅格地图（0=可通行，1=障碍）
            start: 起点
            goal: 终点
        
        Returns:
            路径坐标列表
        """
        height, width = grid.shape
        
        logger.debug(f"[A*] 开始路径规划: grid_size=({width}, {height}), start={start}, goal={goal}")
        
        # 检查边界
        if (start[0] < 0 or start[0] >= width or start[1] < 0 or start[1] >= height or
            goal[0] < 0 or goal[0] >= width or goal[1] < 0 or goal[1] >= height):
            error_msg = f"起点或终点超出地图范围: start={start}, goal={goal}, grid_size=({width}, {height})"
            logger.warning(error_msg)
        
        # 使用公共helper修正起点和终点
        original_start, original_goal = start, goal
        adjusted = _adjust_start_goal_for_obstacle_astar(start, goal, grid, max_radius=100)
        if adjusted is None:
            error_msg = f"起点或终点位于障碍物上且无法找到可通行区域: start={start}, goal={goal}"
            logger.error(error_msg)
        
        new_start, new_goal = adjusted
        if new_start != original_start:
            logger.warning(f"起点位于障碍物上: {original_start}，尝试找到最近的可通行区域")
            logger.info(f"起点已调整: {original_start} -> {new_start}")
            start = new_start
        
        if new_goal != original_goal:
            logger.warning(f"终点位于障碍物上: {original_goal}，尝试找到最近的可通行区域")
            logger.info(f"终点已调整: {original_goal} -> {new_goal}")
            goal = new_goal
        
        # 如果起点就是终点
        if start == goal:
            logger.debug("[A*] 起点和终点相同，返回单点路径")
            return [start]
        
        # 优先队列：(f_score, g_score, current, came_from)
        open_set = [(0, 0, start, None)]
        came_from = {}
        g_score = {start: 0}
        nodes_explored = 0
        
        logger.debug(f"[A*] 开始搜索，起点: {start}, 终点: {goal}")
        
        while open_set:
            _, current_g, current, came_from_pos = heapq.heappop(open_set)
            nodes_explored += 1
            
            # 到达终点
            if current == goal:
                # 重建路径
                path = [goal]
                pos = current
                while pos in came_from:
                    pos = came_from[pos]
                    path.append(pos)
                path.reverse()
                logger.info(f"[A*] 路径规划成功: 路径长度={len(path)}, 探索节点数={nodes_explored}")
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
        error_msg = f"无法找到从起点到终点的路径: start={start}, goal={goal}, 探索节点数={nodes_explored}"
        logger.warning(error_msg)
    
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

    
Coord = Tuple[int, int]  # (x, y)


def _find_nearest_free(
    grid: np.ndarray,
    pos: Tuple[int, int],
    max_radius: int,
    free_predicate: Callable[[float], bool],
) -> Optional[Tuple[int, int]]:
    """
    使用BFS找到最近的可通行点（通用helper）
    
    Args:
        grid: 栅格地图（可以是obstacle_map或cost_map）
        pos: 初始位置 (x, y)
        max_radius: 最大搜索半径（栅格单位）
        free_predicate: 判断某个格子是否可通行的函数，输入为grid[y, x]的值
    
    Returns:
        找到的可通行点 (x, y)，找不到则返回 None
    """
    h, w = grid.shape
    x, y = pos
    
    # 检查边界
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    
    # 如果当前位置就是可通行的
    if free_predicate(grid[y, x]):
        return pos
    
    # BFS搜索（4邻域）
    queue = deque([(x, y, 0)])
    visited = {(x, y)}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while queue:
        cx, cy, dist = queue.popleft()
        
        # 超过最大半径
        if dist >= max_radius:
            continue
        
        # 检查四个方向
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # 检查边界
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            
            # 如果已经访问过
            if (nx, ny) in visited:
                continue
            
            visited.add((nx, ny))
            
            # 如果找到可通行区域
            if free_predicate(grid[ny, nx]):
                logger.debug(f"找到最近可通行区域: ({nx}, {ny}), 距离: {dist + 1}")
                return (nx, ny)
            
            # 继续搜索
            queue.append((nx, ny, dist + 1))
    
    logger.warning(f"在半径 {max_radius} 内未找到可通行区域")
    return None


def _adjust_start_goal_for_cost_astar(
    start: Coord,
    goal: Coord,
    cost_map: np.ndarray,
    max_radius: int = 36,
) -> Optional[Tuple[Coord, Coord]]:
    """
    为cost_map A*修正起点和终点
    
    Args:
        start: 原始起点 (x, y)
        goal: 原始终点 (x, y)
        cost_map: cost_map数组，障碍为np.inf
        max_radius: 最大搜索半径
    
    Returns:
        修正后的(start, goal)，如果修正失败则返回None
    """
    def is_free_cost(v: float) -> bool:
        return np.isfinite(v)
    
    new_start = _find_nearest_free(cost_map, start, max_radius, is_free_cost)
    if new_start is None:
        return None
    
    new_goal = _find_nearest_free(cost_map, goal, max_radius, is_free_cost)
    if new_goal is None:
        return None
    
    return (new_start, new_goal)


def _adjust_start_goal_for_obstacle_astar(
    start: Coord,
    goal: Coord,
    obstacle_map: np.ndarray,
    max_radius: int = 15,
) -> Optional[Tuple[Coord, Coord]]:
    """
    为obstacle_map A*修正起点和终点
    
    Args:
        start: 原始起点 (x, y)
        goal: 原始终点 (x, y)
        obstacle_map: obstacle_map数组，0=可通行，1=障碍
        max_radius: 最大搜索半径
    
    Returns:
        修正后的(start, goal)，如果修正失败则返回None
    """
    def is_free_obstacle(v: float) -> bool:
        return v == 0
    
    new_start = _find_nearest_free(obstacle_map, start, max_radius, is_free_obstacle)
    if new_start is None:
        return None
    
    new_goal = _find_nearest_free(obstacle_map, goal, max_radius, is_free_obstacle)
    if new_goal is None:
        return None
    
    return (new_start, new_goal)


def astar_with_cost(
    cost_map: np.ndarray,
    start: Coord,
    goal: Coord,
    max_adjust_radius: int = 12,
) -> List[Coord]:
    """
    在 cost_map 上做 A*:
        - cost_map[y, x] 为移动代价，障碍为 np.inf

    Args:
        cost_map: HxW float32数组，移动代价（障碍为np.inf）
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)
        max_adjust_radius: 起点/终点修正的最大搜索半径

    Returns:
        path: [(x, y), ...]，从 start 到 goal，找不到则 []
    """
    # 预处理：修正起点和终点
    adjusted = _adjust_start_goal_for_cost_astar(start, goal, cost_map, max_adjust_radius)
    if adjusted is None:
        logger.warning(
            f"cost_map A*: 起点/终点附近找不到可通行区域，规划失败。"
            f"start={start}, goal={goal}, max_radius={max_adjust_radius}"
        )
        return []
    
    new_start, new_goal = adjusted
    original_start, original_goal = start, goal
    
    # 记录修正信息
    if new_start != original_start:
        logger.info(f"cost_map A*: 起点 {original_start} 在障碍上，调整为 {new_start}")
    if new_goal != original_goal:
        logger.info(f"cost_map A*: 终点 {original_goal} 在障碍上，调整为 {new_goal}")
    
    # 使用修正后的起点和终点
    start = new_start
    goal = new_goal
    
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

