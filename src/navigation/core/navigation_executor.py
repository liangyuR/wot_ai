#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三阶段导航执行：路径跟随 + 局部避障 + 在线修正"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import math
import time

from loguru import logger

from ..common.constants import (
    DEFAULT_MOVE_DURATION,
    DEFAULT_MOVE_SPEED,
    DEFAULT_MINIMAP_HEIGHT,
    DEFAULT_MINIMAP_WIDTH,
    DEFAULT_ROTATION_SMOOTH,
    DEFAULT_UPDATE_INTERVAL,
    HEADING_KD,
    HEADING_KI,
    HEADING_KP,
    HEADING_LARGE_TURN,
    HEADING_MAX_OUTPUT,
    HEADING_MEDIUM_TURN,
    HEADING_PWM_PULSE,
    HEADING_REVERSE_ERROR,
    HEADING_REVERSE_PULSE,
    HEADING_REVERSE_THRESHOLD,
    HEADING_SMALL_TURN,
    LOCAL_AVOID_ATTRACT_WEIGHT,
    LOCAL_AVOID_DECAY,
    LOCAL_AVOID_REPEL_RADIUS,
    LOCAL_AVOID_REPEL_WEIGHT,
    MAX_MOVE_DURATION,
    NAV_MIN_PROGRESS_DISTANCE,
    PATH_ADJUST_LOOKAHEAD,
    PATH_DEVIATION_THRESHOLD,
    PATH_LOCAL_REPLAN_THRESHOLD,
    PATH_REPLAN_WINDOW,
    PATH_SAMPLES_PER_SEGMENT,
)
from ..service.control_service import ControlService
from .heading_controller import HeadingControlResult, HeadingController, normalize_angle
from .local_avoidance import PotentialFieldAvoider
from .path_adjuster import OnlinePathAdjuster


class NavigationExecutor:
    """根据阶段三规划执行坦克控制"""

    def __init__(
        self,
        control_service: ControlService,
        move_speed: float = DEFAULT_MOVE_SPEED,
        rotation_smooth: float = DEFAULT_ROTATION_SMOOTH,
    ) -> None:
        if control_service is None:
            raise ValueError("control_service不能为None")
        if move_speed <= 0:
            raise ValueError("move_speed必须是正数")
        if not 0 <= rotation_smooth <= 1:
            raise ValueError("rotation_smooth必须在0-1之间")

        self.control_service_ = control_service
        self.move_speed_ = move_speed
        self.rotation_smooth_ = rotation_smooth

        # 轨迹跟随状态
        self.target_point_offset_ = 5
        self.current_target_idx_ = 0
        self.min_progress_distance_ = NAV_MIN_PROGRESS_DISTANCE

        # 控制器
        self.heading_controller_ = HeadingController(
            kp=HEADING_KP,
            ki=HEADING_KI,
            kd=HEADING_KD,
            max_output=HEADING_MAX_OUTPUT,
        )
        self.avoidance_ = PotentialFieldAvoider(
            attract_weight=LOCAL_AVOID_ATTRACT_WEIGHT,
            repulse_weight=LOCAL_AVOID_REPEL_WEIGHT,
            repel_radius=LOCAL_AVOID_REPEL_RADIUS,
            decay=LOCAL_AVOID_DECAY,
        )
        self.path_adjuster_ = OnlinePathAdjuster(
            deviation_threshold=PATH_DEVIATION_THRESHOLD,
            replan_threshold=PATH_LOCAL_REPLAN_THRESHOLD,
            look_ahead=PATH_ADJUST_LOOKAHEAD,
            segment_window=PATH_REPLAN_WINDOW,
            samples_per_segment=PATH_SAMPLES_PER_SEGMENT,
        )

        # PWM 阈值
        self.small_turn_threshold_ = HEADING_SMALL_TURN
        self.medium_turn_threshold_ = HEADING_MEDIUM_TURN
        self.large_turn_threshold_ = HEADING_LARGE_TURN
        self.micro_pulse_duration_ = HEADING_PWM_PULSE
        self.reverse_command_threshold_ = HEADING_REVERSE_THRESHOLD
        self.reverse_error_threshold_ = HEADING_REVERSE_ERROR
        self.reverse_pulse_duration_ = HEADING_REVERSE_PULSE

    def ExecutePath(
        self,
        path: List[Tuple[int, int]],
        current_pos: Optional[Tuple[float, float]],
        current_heading: Optional[float] = None,
        dynamic_obstacles: Optional[Sequence[Sequence[float]]] = None,
        minimap_size: Optional[Tuple[int, int]] = None,
        update_interval: float = DEFAULT_UPDATE_INTERVAL,
    ) -> None:
        """执行路径并实时纠偏"""

        if not path:
            logger.warning("路径为空，无法执行")
            return
        if update_interval <= 0:
            raise ValueError("update_interval必须是正数")

        dynamic_obstacles = dynamic_obstacles or []
        minimap_size = minimap_size or (DEFAULT_MINIMAP_WIDTH, DEFAULT_MINIMAP_HEIGHT)
        path = list(path)
        self.current_target_idx_ = 0
        last_update = time.time()
        iterations = 0
        max_iterations = max(len(path) * 3, 1)

        while self.current_target_idx_ < len(path) - 1 and iterations < max_iterations:
            iterations += 1

            if current_pos is not None:
                adjustment = self.path_adjuster_.AdjustPath(path, current_pos, self.current_target_idx_)
                if adjustment.replanned:
                    logger.debug(
                        "路径在线重建 deviation={:.2f}, 新长度={}",
                        adjustment.deviation,
                        len(adjustment.path),
                    )
                    path = adjustment.path
                self.current_target_idx_ = min(adjustment.target_idx, len(path) - 1)

            target_idx = min(self.current_target_idx_ + self.target_point_offset_, len(path) - 1)
            source_point = current_pos if current_pos is not None else path[self.current_target_idx_]
            target_point = path[target_idx]
            desired_vec = (
                target_point[0] - source_point[0],
                target_point[1] - source_point[1],
            )

            avoidance_output = None
            if current_pos is not None:
                avoidance_output = self.avoidance_.Compute(
                    current_pos,
                    target_point,
                    obstacles=dynamic_obstacles,
                    minimap_size=minimap_size,
                )
            nav_vec = avoidance_output.vector if avoidance_output is not None else desired_vec
            desired_heading = math.atan2(nav_vec[1], nav_vec[0])

            if current_pos is None or current_heading is None:
                self.RotateToward(target_point, source_point, current_heading)
            else:
                now = time.time()
                dt = max(now - last_update, 1e-3)
                heading_result = self.heading_controller_.Update(current_heading, desired_heading, dt)
                self._apply_heading_command(heading_result)
                current_heading = normalize_angle(current_heading + heading_result.command)
                last_update = now

            travel_distance = math.hypot(desired_vec[0], desired_vec[1])
            duration = min(
                max(travel_distance / 100.0 * self.move_speed_, DEFAULT_MOVE_DURATION * 0.5),
                MAX_MOVE_DURATION,
            )
            self.MoveForward(duration)

            if current_pos is not None:
                current_pos = target_point
            self.current_target_idx_ = target_idx

            if travel_distance < self.min_progress_distance_:
                self.current_target_idx_ = min(self.current_target_idx_ + 1, len(path) - 1)

            time.sleep(update_interval)

        if iterations >= max_iterations:
            logger.warning("导航执行提前结束：迭代次数达到上限")

    def RotateToward(
        self,
        target_pos: Tuple[float, float],
        current_pos: Tuple[float, float],
        current_heading: Optional[float],
    ) -> None:
        """转向目标位置（回退路径）"""

        self.control_service_.RotateToward(target_pos, current_pos, current_heading)

    def MoveForward(self, duration: float) -> None:
        """前进"""

        self.control_service_.MoveForward(duration)

    def EnsureMovingForward(self) -> None:
        """持续按下 W"""

        self.control_service_.StartForward()

    def StopMoving(self) -> None:
        """停止前进"""

        self.control_service_.StopForward()
        logger.debug("停止前进")

    def Stop(self) -> None:
        """释放所有按键"""

        self.StopMoving()
        self.control_service_.Stop()

    def _apply_heading_command(self, result: HeadingControlResult) -> None:
        """将方向控制量转换为 A/D PWM"""

        signal = result.command
        magnitude = abs(signal)

        if magnitude < self.small_turn_threshold_:
            self.control_service_.StopLeft()
            self.control_service_.StopRight()
            return

        pulse = (
            self.micro_pulse_duration_
            if magnitude < self.medium_turn_threshold_
            else self.micro_pulse_duration_ * 1.5
        )

        if signal > 0:
            self.control_service_.StopLeft()
            if magnitude < self.large_turn_threshold_:
                self.control_service_.TapKey('d', pulse)
                self.control_service_.StopRight()
            else:
                self.control_service_.StartRight()
        else:
            self.control_service_.StopRight()
            if magnitude < self.large_turn_threshold_:
                self.control_service_.TapKey('a', pulse)
                self.control_service_.StopLeft()
            else:
                self.control_service_.StartLeft()

        if (
            magnitude > self.reverse_command_threshold_
            and abs(result.error) > self.reverse_error_threshold_
        ):
            self.control_service_.TapKey('s', self.reverse_pulse_duration_)

