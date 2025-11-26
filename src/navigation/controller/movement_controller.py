from __future__ import annotations

import math
from loguru import logger

from src.navigation.controller.move_executor import MovementCommand
from src.navigation.controller.move_executor import MoveExecutor
from src.navigation.nav_runtime.path_follower_wrapper import FollowResult

class MovementController:
    """运动决策层：根据当前位置 / 朝向 / 目标点，算出这一帧的 MovementCommand。

    不直接操作键盘，方便单元测试。
    输出为连续控制量，交给 MoveExecutor 做按键映射。
    """

    def __init__(
        self,
        angle_dead_zone_deg: float = 3.0,
        angle_slow_turn_deg: float = 15.0,
        distance_stop_threshold: float = 5.0,
        slow_down_distance: float = 30.0,
        max_forward_speed: float = 1.0,
        min_forward_factor: float = 0.3,
        large_angle_threshold_deg: float = 60.0,
        large_angle_speed_reduction: float = 0.5,
        corridor_ref_width: float = 40.0, # max_lateral_error
        k_lat_normal: float = 0.3,
        k_lat_edge: float = 0.5,
        k_lat_recenter: float = 0.8,
        straight_angle_enter_deg: float = 6.0,
        straight_angle_exit_deg: float = 10.0,
        straight_lat_enter: float = 25.0,
        straight_lat_exit: float = 35.0,
        edge_speed_reduction: float = 0.85,
        recenter_speed_reduction: float = 0.6,
        debug_log_interval: int = 30,
    ) -> None:
        """初始化控制参数。

        Args:
            angle_dead_zone_deg: 朝向误差小于该角度时认为对准，转向输出趋近于 0
            angle_slow_turn_deg: 误差大于该角度时优先转向，前进速度线性衰减
            distance_stop_threshold: 距离目标小于该值时停止前进
            slow_down_distance: 开始减速的距离（大于 stop_threshold）
            max_forward_speed: 最大前进速度（映射到 MovementCommand.forward，通常在 [0, 1]）
            min_forward_factor: 最小前进因子，确保大角度时仍能保持一定前进速度
            large_angle_threshold_deg: 大角度阈值，超过此角度时进行额外的速度衰减
            large_angle_speed_reduction: 大角度时的速度衰减系数
        """
        if slow_down_distance < distance_stop_threshold:
            slow_down_distance = distance_stop_threshold
        if corridor_ref_width <= 0.0:
            corridor_ref_width = 1.0
        if straight_angle_exit_deg < straight_angle_enter_deg:
            straight_angle_exit_deg = straight_angle_enter_deg + 1.0
        if straight_lat_exit < straight_lat_enter:
            straight_lat_exit = straight_lat_enter + 1.0

        self.angle_dead_zone = math.radians(max(0.0, angle_dead_zone_deg))
        self.angle_slow_turn = math.radians(max(1e-3, angle_slow_turn_deg))
        self.dist_stop = max(0.0, distance_stop_threshold)
        self.dist_slow = max(self.dist_stop, slow_down_distance)
        self.max_forward_speed = max(0.0, max_forward_speed)
        self.min_forward_factor = max(0.0, min(1.0, min_forward_factor))
        self.large_angle_threshold = math.radians(max(0.0, large_angle_threshold_deg))
        self.large_angle_speed_reduction = max(0.0, min(1.0, large_angle_speed_reduction))
        self._corridor_ref_width = corridor_ref_width
        self._k_lat_normal = k_lat_normal
        self._k_lat_edge = k_lat_edge
        self._k_lat_recenter = k_lat_recenter
        self._straight_angle_enter = math.radians(max(0.0, straight_angle_enter_deg))
        self._straight_angle_exit = math.radians(max(0.0, straight_angle_exit_deg))
        self._straight_lat_enter = max(0.0, straight_lat_enter)
        self._straight_lat_exit = max(0.0, straight_lat_exit)
        self._edge_speed_reduction = max(0.1, min(1.0, edge_speed_reduction))
        self._recenter_speed_reduction = max(0.1, min(1.0, recenter_speed_reduction))
        self._log_interval = max(1, debug_log_interval)

        self._straight_mode_active = False
        self._log_counter = 0

    @staticmethod
    def _norm_angle(angle: float) -> float:
        """把角度归一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _update_straight_mode_(self, follow_mode: str, abs_angle: float, abs_lateral: float) -> None:
        """根据当前误差决定是否启用直行模式。"""
        if self._straight_mode_active:
            if (
                abs_angle > self._straight_angle_exit
                or abs_lateral > self._straight_lat_exit
                or follow_mode != "normal"
            ):
                self._straight_mode_active = False
        else:
            if (
                follow_mode == "normal"
                and abs_angle < self._straight_angle_enter
                and abs_lateral < self._straight_lat_enter
            ):
                self._straight_mode_active = True

    def _compute_angle_term_(self, angle_error: float) -> float:
        """非线性角度控制项。"""
        abs_error = abs(angle_error)
        if abs_error < self.angle_dead_zone:
            return 0.0
        norm = angle_error / self.angle_slow_turn
        norm = self._clamp(norm, -1.0, 1.0)
        return math.copysign(norm * norm, norm)

    def _compute_lateral_term_(self, lateral_error: float, mode: str) -> float:
        """横向纠偏控制项。"""
        lat_norm = self._clamp(lateral_error / self._corridor_ref_width, -1.0, 1.0)
        if mode == "recenter":
            gain = self._k_lat_recenter
        elif mode == "corridor_edge":
            gain = self._k_lat_edge
        else:
            gain = self._k_lat_normal
        return gain * lat_norm

    def _compute_forward_speed_(
        self,
        dist_to_target: float,
        abs_angle: float,
        follow: FollowResult,
    ) -> float:
        """根据距离与模式计算前进速度。"""
        if follow.distance_to_goal is not None and follow.distance_to_goal <= self.dist_stop:
            return 0.0
        if dist_to_target <= self.dist_stop or follow.goal_reached:
            return 0.0

        if dist_to_target >= self.dist_slow:
            dist_factor = 1.0
        else:
            denom = max(1e-3, self.dist_slow - self.dist_stop)
            dist_factor = (dist_to_target - self.dist_stop) / denom
            dist_factor = self._clamp(dist_factor, 0.0, 1.0)

        angle_ratio = abs_angle / max(self.angle_slow_turn, 1e-3)
        angle_ratio = self._clamp(angle_ratio, 0.0, 1.0)
        angle_factor = max(self.min_forward_factor, 1.0 - angle_ratio)

        forward_cmd = self.max_forward_speed * dist_factor * angle_factor

        if abs_angle > self.large_angle_threshold:
            forward_cmd *= self.large_angle_speed_reduction

        if follow.mode == "corridor_edge":
            forward_cmd *= self._edge_speed_reduction
        elif follow.mode == "recenter" or follow.need_slowdown:
            forward_cmd *= self._recenter_speed_reduction

        forward_cmd = self._clamp(forward_cmd, 0.0, 1.0)
        return forward_cmd

    def _maybe_log_debug_(
        self,
        *,
        angle_error: float,
        follow: FollowResult,
        turn_cmd: float,
        forward_cmd: float,
    ) -> None:
        """定期输出调试日志，便于调参。"""
        self._log_counter += 1
        if self._log_counter < self._log_interval:
            return
        self._log_counter = 0

        logger.debug(
            "MC mode=%s straight=%s angle_err=%.1fdeg lat=%.1f "
            "forward=%.2f turn=%.2f need_slow=%s",
            follow.mode,
            self._straight_mode_active,
            math.degrees(angle_error),
            follow.lateral_error,
            forward_cmd,
            turn_cmd,
            follow.need_slowdown,
        )

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def decide(
        self,
        current_pos: tuple[float, float],
        heading: float,
        target_pos: tuple[float, float],
    ) -> MovementCommand:
        """兼容旧接口，内部构造 FollowResult。"""
        cx, cy = current_pos
        tx, ty = target_pos
        dist = math.hypot(tx - cx, ty - cy)

        follow = FollowResult(
            target_world=target_pos,
            distance_to_path=0.0,
            lateral_error=0.0,
            distance_to_goal=dist,
            goal_reached=dist < self.dist_stop,
            current_idx=0,
            target_idx_used=0,
            mode="normal",
            need_slowdown=False,
        )
        return self.decide_with_follow_result(
            current_pos=current_pos,
            heading=heading,
            follow=follow,
        )

    def decide_with_follow_result(
        self,
        current_pos: tuple[float, float],
        heading: float,
        follow: FollowResult,
    ) -> MovementCommand:
        """利用 PathFollower 的 FollowResult 进行联动控制。"""
        if self.max_forward_speed <= 0.0 or follow.target_world is None:
            return MovementCommand(forward=0.0, turn=0.0, brake=False)

        cx, cy = current_pos
        tx, ty = follow.target_world
        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)

        distance_to_goal = follow.distance_to_goal if follow.distance_to_goal is not None else dist
        if (
            dist < self.dist_stop
            or distance_to_goal < self.dist_stop
            or follow.goal_reached
        ):
            return MovementCommand(forward=0.0, turn=0.0, brake=False)

        desired_heading = math.atan2(dy, dx)
        angle_error = self._norm_angle(desired_heading - heading)
        abs_angle = abs(angle_error)

        self._update_straight_mode_(
            follow_mode=follow.mode,
            abs_angle=abs_angle,
            abs_lateral=abs(follow.lateral_error),
        )

        if self._straight_mode_active:
            turn_cmd = 0.0
        else:
            angle_term = self._compute_angle_term_(angle_error)
            lat_term = self._compute_lateral_term_(follow.lateral_error, follow.mode)
            turn_cmd = self._clamp(angle_term + lat_term, -1.0, 1.0)

        forward_cmd = self._compute_forward_speed_(dist, abs_angle, follow)

        self._maybe_log_debug_(
            angle_error=angle_error,
            follow=follow,
            turn_cmd=turn_cmd,
            forward_cmd=forward_cmd,
        )

        return MovementCommand(forward=forward_cmd, turn=turn_cmd, brake=False)


if __name__ == "__main__":
    """真实控制演示版 Demo：
    - MovementController 决策
    - MoveExecutor 执行实际按键（W/A/S/D）

    **运行此 demo 会真的按键！**
    建议：
      1. 打开一个空记事本窗口用于观察按键
      2. 不要把鼠标焦点放在游戏窗口上
    """

    import time
    
    time.sleep(5)

    # --- 初始化完整三层结构 ---
    executor = MoveExecutor()     # 控制键盘
    controller = MovementController(         # 输出控制量
        angle_dead_zone_deg=3.0,
        angle_slow_turn_deg=15.0,
        distance_stop_threshold=5.0,
        slow_down_distance=30.0,
        max_forward_speed=1.0,
    )

    print("\n===== MovementController + MoveExecutor Demo =====")
    print("此示例会真实按下 W/A/S/D 键，请确保焦点不在游戏窗口！")
    print("建议打开记事本观察按键效果。")

    # Demo 环境参数
    current_pos = (100.0, 100.0)
    heading = math.radians(0.0)
    target_pos = (200.0, 130.0)

    for i in range(40):
        cmd = controller.decide(
            current_pos=current_pos,
            heading=heading,
            target_pos=target_pos,
        )

        print(
            f"Frame {i:02d}: forward={cmd.forward:.3f}, turn={cmd.turn:.3f}, brake={cmd.brake}"
        )

        # 执行按键
        executor.apply_command(cmd)

        # --- 模拟简单运动 ---
        current_pos = (
            current_pos[0] + cmd.forward * math.cos(heading) * 3.0,
            current_pos[1] + cmd.forward * math.sin(heading) * 3.0,
        )
        heading += cmd.turn * 0.06

        time.sleep(0.05)

    print("演示结束，释放所有按键...")
    executor.stop_all()
    print("===== Demo End =====")