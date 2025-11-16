#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""局部避障：轻量势场模型"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import math


Vector2D = Tuple[float, float]


def _normalize(vec: Vector2D) -> Vector2D:
    norm = math.hypot(vec[0], vec[1])
    if norm == 0:
        return (0.0, 0.0)
    return (vec[0] / norm, vec[1] / norm)


def _scale(vec: Vector2D, scale: float) -> Vector2D:
    return (vec[0] * scale, vec[1] * scale)


def _add(v1: Vector2D, v2: Vector2D) -> Vector2D:
    return (v1[0] + v2[0], v1[1] + v2[1])


@dataclass
class AvoidanceResult:
    vector: Vector2D
    attraction: Vector2D
    repulsion: Vector2D
    obstacle_count: int


class PotentialFieldAvoider:
    """结合吸引/排斥力的轻量势场避障"""

    def __init__(
        self,
        attract_weight: float,
        repulse_weight: float,
        repel_radius: float,
        decay: float,
        min_distance: float = 1.0,
    ) -> None:
        if attract_weight <= 0:
            raise ValueError("attract_weight 必须为正数")
        if repulse_weight <= 0:
            raise ValueError("repulse_weight 必须为正数")
        if repel_radius <= 0:
            raise ValueError("repel_radius 必须为正数")
        if not 0 < decay <= 1.0:
            raise ValueError("decay 需要在 (0, 1] 内")

        self.attract_weight_ = attract_weight
        self.repulse_weight_ = repulse_weight
        self.repel_radius_ = repel_radius
        self.decay_ = decay
        self.min_distance_ = min_distance

    def Compute(
        self,
        current_pos: Optional[Vector2D],
        target_pos: Optional[Vector2D],
        obstacles: Optional[Sequence[Sequence[float]]] = None,
        minimap_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[AvoidanceResult]:
        if current_pos is None or target_pos is None:
            return None

        obstacles = obstacles or []
        base_vec = (target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
        attraction = _scale(_normalize(base_vec), self.attract_weight_)

        repulsion = (0.0, 0.0)
        count = 0
        for obstacle in obstacles:
            if len(obstacle) < 4:
                continue
            cx = 0.5 * (obstacle[0] + obstacle[2])
            cy = 0.5 * (obstacle[1] + obstacle[3])
            dx = current_pos[0] - cx
            dy = current_pos[1] - cy
            dist = math.hypot(dx, dy)
            if dist < self.min_distance_:
                dist = self.min_distance_
            if dist > self.repel_radius_:
                continue

            strength = self.repulse_weight_ / (dist * dist)
            direction = _normalize((dx, dy))
            repulsion = _add(repulsion, _scale(direction, strength))
            count += 1

        if minimap_size is not None:
            w, h = minimap_size
            margin = self.repel_radius_ * 0.5
            if current_pos[0] < margin:
                repulsion = _add(repulsion, (self.repulse_weight_ / (margin * margin), 0.0))
                count += 1
            if current_pos[0] > w - margin:
                repulsion = _add(repulsion, (-self.repulse_weight_ / (margin * margin), 0.0))
                count += 1
            if current_pos[1] < margin:
                repulsion = _add(repulsion, (0.0, self.repulse_weight_ / (margin * margin)))
                count += 1
            if current_pos[1] > h - margin:
                repulsion = _add(repulsion, (0.0, -self.repulse_weight_ / (margin * margin)))
                count += 1

        combined = _add(attraction, _scale(repulsion, self.decay_))
        return AvoidanceResult(
            vector=combined,
            attraction=attraction,
            repulsion=repulsion,
            obstacle_count=count,
        )

