import time
from typing import Optional
from loguru import logger

from src.navigation.controller.movement_controller import MovementController
from src.navigation.controller.move_executor import MoveExecutor, MovementCommand
from src.navigation.nav_runtime.path_follower_wrapper import FollowResult
from src.utils.global_path import GetGlobalConfig

class MovementService:
    """运动控制层 Facade
    对 runtime 暴露简单的 goto/stop 接口，内部组合：
    - MovementController（决策层）
    - MoveExecutor（键盘执行层）
    """

    def __init__(self) -> None:
        config = GetGlobalConfig()
        ctrl_cfg = config.control
        
        missing_det_timeout = getattr(ctrl_cfg, "missing_det_timeout", 1.5)
        self.controller = MovementController(
            angle_dead_zone_deg=ctrl_cfg.angle_dead_zone_deg,
            angle_slow_turn_deg=ctrl_cfg.angle_slow_turn_deg,
            distance_stop_threshold=ctrl_cfg.distance_stop_threshold,
            slow_down_distance=ctrl_cfg.slow_down_distance,
            max_forward_speed=ctrl_cfg.max_forward_speed,
            min_forward_factor=ctrl_cfg.min_forward_factor,
            large_angle_threshold_deg=ctrl_cfg.large_angle_threshold_deg,
            large_angle_speed_reduction=ctrl_cfg.large_angle_speed_reduction,
            corridor_ref_width=ctrl_cfg.corridor_ref_width,
            k_lat_normal=ctrl_cfg.k_lat_normal,
            k_lat_edge=ctrl_cfg.k_lat_edge,
            k_lat_recenter=ctrl_cfg.k_lat_recenter,
            straight_angle_enter_deg=ctrl_cfg.straight_angle_enter_deg,
            straight_angle_exit_deg=ctrl_cfg.straight_angle_exit_deg,
            straight_lat_enter=ctrl_cfg.straight_lat_enter,
            straight_lat_exit=ctrl_cfg.straight_lat_exit,
            edge_speed_reduction=ctrl_cfg.edge_speed_reduction,
            recenter_speed_reduction=ctrl_cfg.recenter_speed_reduction,
            debug_log_interval=ctrl_cfg.debug_log_interval,
        )
        self.executor = MoveExecutor(
            smoothing_alpha_forward=getattr(ctrl_cfg, "smoothing_alpha_forward", 0.2),
            smoothing_alpha_turn=getattr(ctrl_cfg, "smoothing_alpha_turn", 0.6),
            turn_deadzone=ctrl_cfg.turn_deadzone,
            min_hold_time_ms=ctrl_cfg.min_hold_time_ms,
            forward_hysteresis_on=ctrl_cfg.forward_hysteresis_on,
            forward_hysteresis_off=ctrl_cfg.forward_hysteresis_off,
        )

        # 盲走模式状态管理
        self._last_det_ts_: Optional[float] = None
        self._blind_forward_: bool = False
        self._service_start_ts_: Optional[float] = None
        self.missing_det_timeout: float = missing_det_timeout

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

    def forward(self, speed: float = 1.0) -> None:
        """盲走前进：在无路径信息时，简单向前移动一帧。

        Args:
            speed: 前进速度归一化到 [0, 1]。
        """
        clamped_speed = max(0.0, min(1.0, speed))
        cmd = MovementCommand(forward=clamped_speed, turn=0.0, brake=False)
        self.executor.apply_command(cmd)

    def update_detection_status(self, has_detection: bool) -> None:
        """更新检测结果状态，用于盲走模式判断。

        Args:
            has_detection: 当前是否有有效的检测结果（self_pos 不为 None）。
        """
        now = time.perf_counter()
        
        # 记录服务启动时间（首次调用时）
        if self._service_start_ts_ is None:
            self._service_start_ts_ = now

        if has_detection:
            # 有检测结果：更新时间戳，如果之前在盲走模式则退出
            if self._blind_forward_:
                logger.info("检测结果恢复，退出盲走模式并停车")
                self.stop()
                self._blind_forward_ = False
            self._last_det_ts_ = now
        # 如果没有检测结果，时间戳保持不变，由 tick_blind_forward 处理

    def tick_blind_forward(self) -> bool:
        """盲走模式 tick：根据检测结果丢失情况决定是否进入盲走模式。

        应该在控制循环的每一帧调用，当没有检测结果时调用此方法。

        Returns:
            True 表示应该继续盲走（已进入盲走模式），False 表示应该停车等待。
        """
        now = time.perf_counter()

        # 从未成功检测过：出生位置可能被UI遮挡，需要移动才能让检测器看到
        if self._last_det_ts_ is None:
            elapsed_since_start = now - self._service_start_ts_
            if elapsed_since_start < self.missing_det_timeout:
                # 短暂等待，给检测器一些时间
                return False

            # 超过初始等待时间：进入盲走模式
            if not self._blind_forward_:
                logger.info(
                    f"启动后 {elapsed_since_start:.2f}s 仍未检测到位置，"
                    f"可能被UI遮挡，进入盲走模式（持续前进）"
                )
                self._blind_forward_ = True

            # 进入盲走模式，持续前进
            self.forward()
            return True

        # 曾经检测成功过，但后来丢失了
        missing_dt = now - self._last_det_ts_
        if missing_dt < self.missing_det_timeout:
            # 短暂丢失：停车等待检测恢复
            if self._blind_forward_:
                logger.info("检测短暂丢失，退出盲走模式并停车")
                self.stop()
                self._blind_forward_ = False
            return False

        # 超过超时时间：进入盲走模式，持续向前移动
        if not self._blind_forward_:
            logger.info(
                f"检测结果已丢失 {missing_dt:.2f}s，超过阈值 "
                f"{self.missing_det_timeout:.2f}s，进入盲走模式（持续前进）"
            )
            self._blind_forward_ = True

        # 简单前进一帧，由 MoveExecutor 负责按键保持和滞回
        self.forward()
        return True