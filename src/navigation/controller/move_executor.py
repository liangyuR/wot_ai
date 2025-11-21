from __future__ import annotations
from dataclasses import dataclass
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

    def __init__(self) -> None:
        self.key_press_manager = KeyPressManager()
        self._state = _KeyState()

        # 调参阈值：小于这个值就认为是 0
        self._forward_deadzone = 0.2
        self._turn_deadzone = 0.2

    # -------- public API (给 movement_controller 调用) --------

    def apply_command(self, cmd: MovementCommand) -> None:
        """根据 MovementCommand 更新键盘按键状态。

        由 movement_controller 在固定周期调用，如 20~30Hz。
        """
        if cmd.brake:
            self.stop_all()
            return

        target_forward = self._quantize_axis(cmd.forward, self._forward_deadzone)
        target_turn = self._quantize_axis(cmd.turn, self._turn_deadzone)

        # 前后方向
        if target_forward != self._state.forward:
            self._update_forward(target_forward)

        # 左右方向
        if target_turn != self._state.turn:
            self._update_turn(target_turn)

    def stop_all(self) -> None:
        """释放所有移动相关按键，并清空内部状态"""
        self.key_press_manager.stop_all()
        self._state = _KeyState()

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
