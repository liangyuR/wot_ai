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
        target_point_offset: int,
        goal_arrival_threshold: float,
    ):
        """
        Args:
            follower: 你现有的 PathFollower 实例
            deviation_tolerance: 路径偏离容忍度（px）
            target_point_offset: 前瞻点偏移量（索引）
            goal_arrival_threshold: 认为“到达终点”的距离阈值（px）
        """
        self._follower = PathFollower()
        self._dev_tol = deviation_tolerance
        self._offset = max(0, target_point_offset)
        self._goal_th = goal_arrival_threshold

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

        new_current_idx = max(current_target_idx, idx)

        if deviation > self._dev_tol * 2:
            logger.warning(f"路径偏离较大: {deviation:.1f}px (tol={self._dev_tol})")

        # 2. 终点判断
        goal = path_world[-1]
        dx_g = goal[0] - current_pos[0]
        dy_g = goal[1] - current_pos[1]
        dist_goal = math.hypot(dx_g, dy_g)
        goal_reached = dist_goal < self._goal_th

        if goal_reached:
            return None, deviation, dist_goal, True, new_current_idx, new_current_idx

        # 3. 前瞻目标点
        target_idx = min(new_current_idx + self._offset, len(path_world) - 1)
        target_world = path_world[target_idx]

        return target_world, deviation, dist_goal, False, new_current_idx, target_idx
