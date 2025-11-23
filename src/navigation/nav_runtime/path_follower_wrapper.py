#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Tuple, Optional
from loguru import logger

from src.navigation.core.path_follower import PathFollower


class PathFollowerWrapper:
    """
    封装 PathFollower 的路径跟随逻辑：
    - 最近路径点查找
    - 偏离计算
    - 前瞻目标点选择
    - 到达终点判断
    """

    def __init__(
        self,
        deviation_tolerance: float,
        goal_arrival_threshold: float,
        max_lateral_error: float = 80.0,
        lookahead_distance: float = 60.0,
        waypoint_switch_radius: float = 20.0,
    ):
        """
        Args:
            deviation_tolerance: 路径偏离容忍度（px）
            goal_arrival_threshold: 认为"到达终点"的距离阈值（px）
            max_lateral_error: 最大横向误差，定义corridor宽度（px）
            lookahead_distance: 前瞻距离，用于计算胡萝卜点（px）
            waypoint_switch_radius: Waypoint切换半径（px）
        """
        self._follower = PathFollower()
        self._dev_tol = deviation_tolerance
        self._goal_th = goal_arrival_threshold
        self._max_lateral_error = max_lateral_error
        self._lookahead_dist = lookahead_distance
        self._waypoint_switch_radius = waypoint_switch_radius

    def follow(
        self,
        current_pos: Tuple[float, float],
        path_world: List[Tuple[float, float]],
        current_target_idx: int,
    ) -> Tuple[
        Optional[Tuple[float, float]],  # target_world
        float,                          # deviation
        float,                          # distance_to_goal
        bool,                           # goal_reached
        int,                            # new_current_target_idx
        int,                            # target_idx_used
    ]:
        """
        基于当前位置和路径计算下一步目标点及状态信息。

        Returns:
            target_world: 本帧使用的目标点（None 表示不用移动）
            deviation: 当前偏离路径的距离
            distance_to_goal: 距终点距离
            goal_reached: 是否到达终点
            new_current_target_idx: 更新后的当前路径索引
            target_idx_used: 用于控制的目标点索引（含前瞻）
        """
        if not path_world or len(path_world) < 2:
            # 没有路径或路径太短，认为已经到达
            return None, 0.0, 0.0, True, 0, 0

        # 1. 最近路径点
        search_start = max(0, current_target_idx - 5)
        idx, nearest_pt, deviation = self._follower.find_nearest_point(
            current_pos, path_world, search_start
        )

        # 计算横向偏差（lateral error）
        # 使用最近点到当前位置的距离作为横向偏差
        lateral_error = deviation

        # 2. Waypoint切换：当距离当前waypoint小于切换半径时，切换到下一个
        new_current_idx = max(current_target_idx, idx)
        if new_current_idx < len(path_world) - 1:
            current_waypoint = path_world[new_current_idx]
            dist_to_waypoint = math.hypot(
                current_waypoint[0] - current_pos[0],
                current_waypoint[1] - current_pos[1]
            )
            if dist_to_waypoint < self._waypoint_switch_radius:
                new_current_idx = min(new_current_idx + 1, len(path_world) - 1)

        if deviation > self._dev_tol * 2:
            logger.warning(f"路径偏离较大: {deviation:.1f}px (tol={self._dev_tol})")

        # 3. 终点判断
        goal = path_world[-1]
        dx_g = goal[0] - current_pos[0]
        dy_g = goal[1] - current_pos[1]
        dist_goal = math.hypot(dx_g, dy_g)
        goal_reached = dist_goal < self._goal_th

        if goal_reached:
            return None, deviation, dist_goal, True, new_current_idx, new_current_idx

        # 4. 计算前瞻目标点（胡萝卜点）
        # 从当前索引开始，累计距离直到达到lookahead_distance
        carrot_idx = new_current_idx
        accumulated_dist = 0.0
        
        for i in range(new_current_idx, len(path_world) - 1):
            p1 = path_world[i]
            p2 = path_world[i + 1]
            seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            
            if accumulated_dist + seg_dist >= self._lookahead_dist:
                # 在当前线段上插值
                remaining = self._lookahead_dist - accumulated_dist
                ratio = remaining / seg_dist if seg_dist > 0 else 0.0
                carrot_x = p1[0] + ratio * (p2[0] - p1[0])
                carrot_y = p1[1] + ratio * (p2[1] - p1[1])
                target_world = (carrot_x, carrot_y)
                carrot_idx = i + 1
                break
            
            accumulated_dist += seg_dist
            carrot_idx = i + 1
        
        # 如果累计距离未达到lookahead_distance，使用最后一个点
        if carrot_idx >= len(path_world) - 1:
            target_world = path_world[-1]
            carrot_idx = len(path_world) - 1

        # 5. Corridor判断：根据横向误差决定控制策略
        # 这里返回的target_world已经考虑了lookahead，corridor逻辑可以在MovementController中进一步处理
        # 或者在这里根据lateral_error调整target_world的位置
        if abs(lateral_error) >= self._max_lateral_error:
            # 偏离走廊，目标点更偏向路径中心
            # 这里可以进一步优化，暂时先返回lookahead点
            logger.debug(f"偏离走廊: lateral_error={lateral_error:.1f}px (max={self._max_lateral_error})")

        return target_world, deviation, dist_goal, False, new_current_idx, carrot_idx
