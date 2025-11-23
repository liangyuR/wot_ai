from __future__ import annotations
from dataclasses import dataclass
import time
from loguru import logger
from src.navigation.controller.keyboard_manager import KeyPressManager

@dataclass
class MovementCommand:
    """高层运动指令，由 movement_controller 产生。

    forward: -1.0 ~ 1.0  负数后退，正数前进，0 停车
    turn:    -1.0 ~ 1.0  负数左转，正数右转，0 不转向
    """

    forward: float = 0.0
    turn: float = 0.0
    brake: bool = False  # 急停信号，优先级最高


# ===================== Command → Key mapping =====================


@dataclass
class _KeyState:
    forward: int = 0  # -1 = S, 0 = none, 1 = W
    turn: int = 0  # -1 = A, 0 = none, 1 = D


class MoveExecutor:
    """移动执行器

    负责：
    - 根据 MovementCommand 控制 WASD
    - 维护当前按键状态，避免重复 press/release
    - 提供统一的 stop_all 接口
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        turn_deadzone: float = 0.12,
        min_hold_time_ms: float = 100.0,
        forward_hysteresis_on: float = 0.35,
        forward_hysteresis_off: float = 0.08,
    ) -> None:
        self.key_press_manager = KeyPressManager()
        self._state = _KeyState()

        # 平滑滤波参数
        self._smooth_forward: float = 0.0
        self._smooth_turn: float = 0.0
        self._smoothing_alpha: float = max(0.0, min(1.0, smoothing_alpha))

        # 调参阈值：小于这个值就认为是 0
        self._turn_deadzone = turn_deadzone

        # 滞回参数
        self._forward_hysteresis_on = forward_hysteresis_on
        self._forward_hysteresis_off = forward_hysteresis_off

        # 最小按键保持时间（秒）
        self._min_hold_time = min_hold_time_ms / 1000.0
        self._last_forward_change: float = 0.0
        self._last_turn_change: float = 0.0

    # -------- public API (给 movement_controller 调用) --------

    def apply_command(self, cmd: MovementCommand) -> None:
        """根据 MovementCommand 更新键盘按键状态。

        由 movement_controller 在固定周期调用，如 20~30Hz。
        """
        if cmd.brake:
            self.stop_all()
            return

        # 一阶指数平滑滤波
        self._smooth_forward += self._smoothing_alpha * (cmd.forward - self._smooth_forward)
        self._smooth_turn += self._smoothing_alpha * (cmd.turn - self._smooth_turn)

        # 使用平滑后的值进行量化
        target_forward = self._quantize_forward_with_hysteresis(self._smooth_forward)
        target_turn = self._quantize_axis(self._smooth_turn, self._turn_deadzone)

        # 前后方向（带最小保持时间检查）
        if target_forward != self._state.forward:
            now = time.perf_counter()
            if now - self._last_forward_change >= self._min_hold_time:
                self._update_forward(target_forward)
                self._last_forward_change = now

        # 左右方向（带最小保持时间检查）
        if target_turn != self._state.turn:
            now = time.perf_counter()
            if now - self._last_turn_change >= self._min_hold_time:
                self._update_turn(target_turn)
                self._last_turn_change = now

        # 调试日志
        logger.debug(
            f"cmd: fwd={cmd.forward:.3f}, turn={cmd.turn:.3f} | "
            f"smooth: fwd={self._smooth_forward:.3f}, turn={self._smooth_turn:.3f} | "
            f"state: fwd={self._state.forward}, turn={self._state.turn}"
        )

    def stop_all(self) -> None:
        """释放所有移动相关按键，并清空内部状态"""
        self.key_press_manager.stop_all()
        self._state = _KeyState()
        self._smooth_forward = 0.0
        self._smooth_turn = 0.0
        self._last_forward_change = 0.0
        self._last_turn_change = 0.0

    # -------- internal helpers: forward/backward --------

    def _update_forward(self, direction: int) -> None:
        """direction: -1 = 后退(S), 0 = 松开, 1 = 前进(W)"""
        # 先根据当前状态松键
        if self._state.forward == 1:
            self.key_press_manager.release_forward()
        elif self._state.forward == -1:
            self.key_press_manager.release_backward()

        # 再按新键
        if direction == 1:
            self.key_press_manager.press_forward()
        elif direction == -1:
            self.key_press_manager.press_backward()

        logger.debug(f"update_forward: {self._state.forward} -> {direction}")
        self._state.forward = direction

    # -------- internal helpers: left/right --------

    def _update_turn(self, direction: int) -> None:
        """direction: -1 = 左转(A), 0 = 松开, 1 = 右转(D)"""
        if self._state.turn == 1:
            self.key_press_manager.release_turn_right()
        elif self._state.turn == -1:
            self.key_press_manager.release_turn_left()

        if direction == 1:
            self.key_press_manager.press_turn_right()
        elif direction == -1:
            self.key_press_manager.press_turn_left()

        logger.debug(f"update_turn: {self._state.turn} -> {direction}")
        self._state.turn = direction

    # -------- utility --------

    def _quantize_forward_with_hysteresis(self, value: float) -> int:
        """前进方向带滞回的量化：

        - 当前状态为0：value > on_thr → 1，value < -on_thr → -1
        - 当前状态为1：value < off_thr → 0，否则保持1
        - 当前状态为-1：value > -off_thr → 0，否则保持-1
        """
        cur = self._state.forward
        if cur == 0:
            if value > self._forward_hysteresis_on:
                return 1
            if value < -self._forward_hysteresis_on:
                return -1
            return 0
        elif cur == 1:
            if value < self._forward_hysteresis_off:
                return 0
            return 1
        else:  # cur == -1
            if value > -self._forward_hysteresis_off:
                return 0
            return -1

    @staticmethod
    def _quantize_axis(value: float, deadzone: float) -> int:
        """连续控制量 → 离散方向：

        - |value| < deadzone      -> 0
        - value >= deadzone       -> 1
        - value <= -deadzone      -> -1
        """
        if value > deadzone:
            return 1
        if value < -deadzone:
            return -1
        return 0
