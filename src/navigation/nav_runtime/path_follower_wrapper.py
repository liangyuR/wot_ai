#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger

@dataclass
class FollowResult:
    """Path follower output shared with MovementController."""

    target_world: Optional[Tuple[float, float]]
    distance_to_path: float
    lateral_error: float
    distance_to_goal: float
    goal_reached: bool
    current_idx: int
    target_idx_used: int
    mode: str
    need_slowdown: bool


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
        inner_corridor: Optional[float] = None,
        edge_lookahead_scale: float = 0.6,
        lookahead_distance: float = 60.0,
        waypoint_switch_radius: float = 20.0,
        recenter_forward_offset: float = 10.0,
    ):
        """
        Args:
            deviation_tolerance: 路径偏离容忍度（px）
            goal_arrival_threshold: 认为"到达终点"的距离阈值（px）
            max_lateral_error: 最大横向误差，定义corridor宽度（px）
            inner_corridor: 内走廊宽度，None 时默认 tolerance 的 3 倍
            edge_lookahead_scale: 走廊边缘前瞻缩放系数
            lookahead_distance: 前瞻距离，用于计算胡萝卜点（px）
            waypoint_switch_radius: Waypoint切换半径（px）
            recenter_forward_offset: recenter 模式沿切线前推距离（px）
        """
        self._dev_tol = deviation_tolerance
        self._goal_th = goal_arrival_threshold
        
        self._max_lateral_error = max_lateral_error
        self._inner_corridor = (
            inner_corridor if inner_corridor is not None else deviation_tolerance * 3.0
        )
        if self._inner_corridor >= self._max_lateral_error:
            logger.warning(
                "inner_corridor(%.1f) >= max_lateral_error(%.1f)，corridor_edge 区间将消失，自动收缩 inner_corridor",
                self._inner_corridor, self._max_lateral_error,
            )
            self._inner_corridor = self._max_lateral_error * 0.7

        self._edge_lookahead_scale = max(0.1, min(edge_lookahead_scale, 1.0))
        self._lookahead_dist = lookahead_distance
        self._waypoint_switch_radius = waypoint_switch_radius
        self._recenter_forward_offset = max(0.0, recenter_forward_offset)

    def follow(
        self,
        current_pos: Tuple[float, float],
        path_world: List[Tuple[float, float]],
        current_target_idx: int,
    ) -> FollowResult:
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
            return FollowResult(
                target_world=None,
                distance_to_path=0.0,
                lateral_error=0.0,
                distance_to_goal=0.0,
                goal_reached=True,
                current_idx=0,
                target_idx_used=0,
                mode="normal",
                need_slowdown=False,
            )

        # 1. 最近路径点
        search_start = max(0, current_target_idx - 5)
        idx, nearest_pt, deviation = self.find_nearest_point(
            current_pos, path_world, search_start
        )

        signed_lateral_error = self._compute_signed_lateral_error_(
            path_world=path_world,
            idx=idx,
            nearest_pt=nearest_pt,
            current_pos=current_pos,
            deviation=deviation,
        )

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
            return FollowResult(
                target_world=None,
                distance_to_path=deviation,
                lateral_error=signed_lateral_error,
                distance_to_goal=dist_goal,
                goal_reached=True,
                current_idx=new_current_idx,
                target_idx_used=new_current_idx,
                mode="normal",
                need_slowdown=False,
            )

        # 4. 计算前瞻目标点（胡萝卜点）
        # 从当前索引开始，累计距离直到达到lookahead_distance
        mode, need_slowdown = self._classify_mode_(signed_lateral_error)
        effective_lookahead = self._select_lookahead_(mode)

        if mode == "recenter":
            target_world, carrot_idx = self._build_recenter_target_(
                idx=idx,
                nearest_pt=nearest_pt,
                path_world=path_world,
            )
        else:
            target_world, carrot_idx = self._compute_carrot_target_(
                start_idx=new_current_idx,
                path_world=path_world,
                lookahead=effective_lookahead,
            )

        logger.debug(
            "PFW idx=%d carrot_idx=%d mode=%s lat_err=%.2f lookahead=%.1f",
            idx,
            carrot_idx,
            mode,
            signed_lateral_error,
            effective_lookahead,
        )

        return FollowResult(
            target_world=target_world,
            distance_to_path=deviation,
            lateral_error=signed_lateral_error,
            distance_to_goal=dist_goal,
            goal_reached=False,
            current_idx=new_current_idx,
            target_idx_used=carrot_idx,
            mode=mode,
            need_slowdown=need_slowdown,
        )

    def _compute_signed_lateral_error_(
        self,
        path_world: List[Tuple[float, float]],
        idx: int,
        nearest_pt: Tuple[float, float],
        current_pos: Tuple[float, float],
        deviation: float,
    ) -> float:
        """使用路径切线和叉积计算有符号横向误差。"""
        tangent = self._estimate_tangent_(path_world, idx)
        dx = current_pos[0] - nearest_pt[0]
        dy = current_pos[1] - nearest_pt[1]
        cross = tangent[0] * dy - tangent[1] * dx
        if cross == 0.0:
            return deviation
        return math.copysign(deviation, cross)

    def _estimate_tangent_(
        self, path_world: List[Tuple[float, float]], idx: int
    ) -> Tuple[float, float]:
        """估算路径切线方向。"""
        if idx < len(path_world) - 1:
            t = (
                path_world[idx + 1][0] - path_world[idx][0],
                path_world[idx + 1][1] - path_world[idx][1],
            )
        elif idx > 0:
            t = (
                path_world[idx][0] - path_world[idx - 1][0],
                path_world[idx][1] - path_world[idx - 1][1],
            )
        else:
            t = (1.0, 0.0)

        norm = math.hypot(t[0], t[1])
        if norm == 0.0:
            return (1.0, 0.0)
        return (t[0] / norm, t[1] / norm)

    def _classify_mode_(self, signed_lateral_error: float) -> Tuple[str, bool]:
        """根据横向误差划分走廊模式。"""
        abs_err = abs(signed_lateral_error)
        if abs_err < self._inner_corridor:
            return "normal", False
        if abs_err < self._max_lateral_error:
            return "corridor_edge", False
        return "recenter", True

    def _select_lookahead_(self, mode: str) -> float:
        """根据模式选择前瞻距离。"""
        if mode == "corridor_edge":
            return self._lookahead_dist * self._edge_lookahead_scale
        if mode == "recenter":
            return 0.0
        return self._lookahead_dist

    def _compute_carrot_target_(
        self,
        start_idx: int,
        path_world: List[Tuple[float, float]],
        lookahead: float,
    ) -> Tuple[Tuple[float, float], int]:
        """沿路径计算前瞻目标点。"""
        carrot_idx = start_idx
        accumulated_dist = 0.0
        target_world: Tuple[float, float] = path_world[start_idx]

        for i in range(start_idx, len(path_world) - 1):
            p1 = path_world[i]
            p2 = path_world[i + 1]
            seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

            if accumulated_dist + seg_dist >= lookahead:
                remaining = lookahead - accumulated_dist
                ratio = remaining / seg_dist if seg_dist > 0 else 0.0
                target_world = (
                    p1[0] + ratio * (p2[0] - p1[0]),
                    p1[1] + ratio * (p2[1] - p1[1]),
                )
                carrot_idx = i + 1
                break

            accumulated_dist += seg_dist
            carrot_idx = i + 1
            target_world = p2

        if carrot_idx >= len(path_world) - 1:
            target_world = path_world[-1]
            carrot_idx = len(path_world) - 1

        return target_world, carrot_idx

    def _build_recenter_target_(
        self,
        idx: int,
        nearest_pt: Tuple[float, float],
        path_world: List[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], int]:
        """重度偏离时以最近点附近为目标。"""
        tangent = self._estimate_tangent_(path_world, idx)
        target_world = (
            nearest_pt[0] + tangent[0] * self._recenter_forward_offset,
            nearest_pt[1] + tangent[1] * self._recenter_forward_offset,
        )
        return target_world, idx

        
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
        n = len(path_world)
        start_idx = max(0, min(start_idx, n - 1))
        nearest_idx = start_idx
        nearest_point = path_world[start_idx]
        
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