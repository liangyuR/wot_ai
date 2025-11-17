#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方向控制器：基于 PID/PD 的实时转向修正"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-π, π] 区间"""

    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class HeadingControlResult:
    """方向控制输出"""

    error: float
    command: float
    heading_target: float
    heading_now: float


class HeadingController:
    """基于 PID/PD 的方向控制器"""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        max_output: float,
        integral_limit: Optional[float] = None,
    ) -> None:
        if kp < 0 or ki < 0 or kd < 0:
            raise ValueError("PID 系数必须为非负数")
        if max_output <= 0:
            raise ValueError("max_output 必须为正数")

        self.kp_ = kp
        self.ki_ = ki
        self.kd_ = kd
        self.max_output_ = max_output
        self.integral_limit_ = integral_limit
        self.integral_term_: float = 0.0
        self.prev_error_: Optional[float] = None

    def Reset(self) -> None:
        """重置控制器状态"""

        self.integral_term_ = 0.0
        self.prev_error_ = None

    def Update(self, heading_now: float, heading_target: float, dt: float) -> HeadingControlResult:
        """根据当前角度和目标角度计算转向控制量"""

        if dt <= 0:
            dt = 1e-3

        error = normalize_angle(heading_target - heading_now)

        if self.ki_ > 0:
            self.integral_term_ += error * dt
            if self.integral_limit_ is not None:
                self.integral_term_ = max(
                    -self.integral_limit_,
                    min(self.integral_limit_, self.integral_term_),
                )
        else:
            self.integral_term_ = 0.0

        derivative = 0.0
        if self.prev_error_ is not None:
            derivative = (error - self.prev_error_) / dt
        self.prev_error_ = error

        command = self.kp_ * error + self.ki_ * self.integral_term_ + self.kd_ * derivative
        command = max(-self.max_output_, min(self.max_output_, command))

        return HeadingControlResult(
            error=error,
            command=command,
            heading_target=heading_target,
            heading_now=heading_now,
        )

