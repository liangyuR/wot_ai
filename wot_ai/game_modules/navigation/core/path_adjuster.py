#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三阶段：轨迹在线调整"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from .catmull_rom import catmull_rom_interpolate

Point = Tuple[float, float]


def _distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


@dataclass
class PathAdjustmentResult:
    path: List[Tuple[int, int]]
    target_idx: int
    deviation: float
    replanned: bool


class OnlinePathAdjuster:
    """根据偏离程度实时修正路径"""

    def __init__(
        self,
        deviation_threshold: float,
        replan_threshold: float,
        look_ahead: int,
        segment_window: int,
        samples_per_segment: int,
    ) -> None:
        if deviation_threshold <= 0 or replan_threshold <= 0:
            raise ValueError("阈值必须大于 0")
        if replan_threshold < deviation_threshold:
            raise ValueError("replan_threshold 必须大于 deviation_threshold")
        self.deviation_threshold_ = deviation_threshold
        self.replan_threshold_ = replan_threshold
        self.look_ahead_ = max(3, look_ahead)
        self.segment_window_ = max(2, segment_window)
        self.samples_per_segment_ = max(5, samples_per_segment)

    def AdjustPath(
        self,
        path: List[Tuple[int, int]],
        current_pos: Optional[Point],
        current_idx: int,
    ) -> PathAdjustmentResult:
        if not path or current_pos is None:
            return PathAdjustmentResult(path, current_idx, 0.0, False)

        nearest_idx, min_dist = self._find_nearest_point(path, current_pos, current_idx)

        replanned = False
        new_path = path
        new_idx = nearest_idx

        if min_dist > self.replan_threshold_ and len(path) - nearest_idx > 2:
            new_path = self._regenerate_segment(path, current_pos, nearest_idx)
            new_idx = max(0, nearest_idx - 1)
            replanned = True
        elif min_dist > self.deviation_threshold_:
            new_idx = nearest_idx

        return PathAdjustmentResult(new_path, new_idx, min_dist, replanned)

    def _find_nearest_point(
        self,
        path: List[Tuple[int, int]],
        current_pos: Point,
        current_idx: int,
    ) -> Tuple[int, float]:
        start = max(0, current_idx - 1)
        end = min(len(path), current_idx + self.look_ahead_)
        min_dist = float("inf")
        nearest_idx = current_idx
        for idx in range(start, end):
            dist = _distance(current_pos, path[idx])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        return nearest_idx, min_dist

    def _regenerate_segment(
        self,
        path: List[Tuple[int, int]],
        current_pos: Point,
        anchor_idx: int,
    ) -> List[Tuple[int, int]]:
        window_end = min(len(path), anchor_idx + self.segment_window_)
        control_points: List[Point] = [current_pos]
        control_points.extend(path[anchor_idx:window_end])
        if len(control_points) < 2:
            return path

        interpolated = catmull_rom_interpolate(
            control_points,
            num_points_per_segment=self.samples_per_segment_,
            alpha=0.5,
        )
        interpolated_int = [(int(round(x)), int(round(y))) for x, y in interpolated]

        return path[:anchor_idx] + interpolated_int + path[window_end:]

