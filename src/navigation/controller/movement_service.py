from src.navigation.controller.movement_controller import MovementController
from src.navigation.controller.move_executor import MoveExecutor, MovementCommand

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
    ) -> None:
        self.controller = MovementController(
            angle_dead_zone_deg=angle_dead_zone_deg,
            angle_slow_turn_deg=angle_slow_turn_deg,
            distance_stop_threshold=distance_stop_threshold,
            slow_down_distance=slow_down_distance,
            max_forward_speed=max_forward_speed,
        )
        self.executor = MoveExecutor()

    # ---- runtime 调用的主要接口 ----

    def goto(
        self,
        *,
        target_pos: tuple[float, float],
        current_pos: tuple[float, float],
        heading: float,
    ) -> None:
        """朝向 target_pos 运动一帧。

        由 runtime 在固定 tick 里反复调用，相当于旧版的 self.move.goto(...)
        """
        cmd = self.controller.decide(
            current_pos=current_pos,
            heading=heading,
            target_pos=target_pos,
        )
        self.executor.apply_command(cmd)

    def stop(self) -> None:
        """停车（释放所有移动相关按键）"""
        self.executor.stop_all()

    def brake(self) -> None:
        """急停：可选接口，当前先等价于 stop。未来可以加特殊逻辑。"""
        self.executor.apply_command(MovementCommand(forward=0.0, turn=0.0, brake=True))
        self.executor.stop_all()
