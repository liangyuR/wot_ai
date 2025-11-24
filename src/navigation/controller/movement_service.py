from src.navigation.controller.movement_controller import MovementController
from src.navigation.controller.move_executor import MoveExecutor, MovementCommand
from src.navigation.nav_runtime.path_follower_wrapper import FollowResult

class MovementService:
    """运动控制层 Facade
    对 runtime 暴露简单的 goto/stop 接口，内部组合：
    - MovementController（决策层）
    - MoveExecutor（键盘执行层）
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
        corridor_ref_width: float = 40.0,
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
        smoothing_alpha: float = 0.3,
        turn_deadzone: float = 0.12,
        min_hold_time_ms: float = 100.0,
        forward_hysteresis_on: float = 0.35,
        forward_hysteresis_off: float = 0.08,
    ) -> None:
        self.controller = MovementController(
            angle_dead_zone_deg=angle_dead_zone_deg,
            angle_slow_turn_deg=angle_slow_turn_deg,
            distance_stop_threshold=distance_stop_threshold,
            slow_down_distance=slow_down_distance,
            max_forward_speed=max_forward_speed,
            min_forward_factor=min_forward_factor,
            large_angle_threshold_deg=large_angle_threshold_deg,
            large_angle_speed_reduction=large_angle_speed_reduction,
            corridor_ref_width=corridor_ref_width,
            k_lat_normal=k_lat_normal,
            k_lat_edge=k_lat_edge,
            k_lat_recenter=k_lat_recenter,
            straight_angle_enter_deg=straight_angle_enter_deg,
            straight_angle_exit_deg=straight_angle_exit_deg,
            straight_lat_enter=straight_lat_enter,
            straight_lat_exit=straight_lat_exit,
            edge_speed_reduction=edge_speed_reduction,
            recenter_speed_reduction=recenter_speed_reduction,
            debug_log_interval=debug_log_interval,
        )
        self.executor = MoveExecutor(
            smoothing_alpha=smoothing_alpha,
            turn_deadzone=turn_deadzone,
            min_hold_time_ms=min_hold_time_ms,
            forward_hysteresis_on=forward_hysteresis_on,
            forward_hysteresis_off=forward_hysteresis_off,
        )

    # ---- runtime 调用的主要接口 ----

    def goto(
        self,
        *,
        follow_result: FollowResult,
        current_pos: tuple[float, float],
        heading: float,
    ) -> None:
        """根据 PathFollower 状态执行运动一帧。"""
        cmd = self.controller.decide_with_follow_result(
            current_pos=current_pos,
            heading=heading,
            follow=follow_result,
        )
        self.executor.apply_command(cmd)

    def stop(self) -> None:
        """停车（释放所有移动相关按键）"""
        self.executor.stop_all()

    def brake(self) -> None:
        """急停：可选接口，当前先等价于 stop。未来可以加特殊逻辑。"""
        self.executor.apply_command(MovementCommand(forward=0.0, turn=0.0, brake=True))
        self.executor.stop_all()
