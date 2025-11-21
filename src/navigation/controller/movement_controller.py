# movement_controller.py

from dataclasses import dataclass
from typing import Optional, Tuple
import math
from loguru import logger


@dataclass
class MovementCommand:
    """这一帧应该处于什么运动状态"""
    forward: bool          # 是否按 W
    turn: int              # -1: 左转(A), 0: 不转, 1: 右转(D)


class MovementController:
    """
    运动决策层：根据当前位置 / 朝向 / 目标点，算出这一帧的 MovementCommand。
    不直接操作键盘，方便单元测试。
    """

    def __init__(
        self,
        angle_dead_zone_deg: float = 3.0,
        angle_slow_turn_deg: float = 15.0,
        distance_stop_threshold: float = 5.0,
    ) -> None:
        """
        Args:
            angle_dead_zone_deg: 朝向误差小于该角度时认为对准，不再转向
            angle_slow_turn_deg: 误差大于该角度时，先纯转向、不前进
            distance_stop_threshold: 距离目标小于该值时停止前进
        """
        self.angle_dead_zone = math.radians(angle_dead_zone_deg)
        self.angle_slow_turn = math.radians(angle_slow_turn_deg)
        self.dist_stop = distance_stop_threshold

    @staticmethod
    def _norm_angle(angle: float) -> float:
        """把角度归一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def decide(
        self,
        current_pos: Tuple[float, float],
        heading: float,
        target_pos: Tuple[float, float],
    ) -> MovementCommand:
        """
        核心决策：往哪个方向转，是否前进。

        Args:
            current_pos: 当前 (x, y) 世界坐标（像素）
            heading: 当前朝向（弧度，0 向右，逆时针为正）
            target_pos: 目标 (x, y) 世界坐标（像素）
        """
        cx, cy = current_pos
        tx, ty = target_pos

        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)

        # 很近就直接停住
        if dist < self.dist_stop:
            return MovementCommand(forward=False, turn=0)

        # 目标方向角
        desired = math.atan2(dy, dx)
        diff = self._norm_angle(desired - heading)

        # 朝向基本对准：保持前进，不转向
        if abs(diff) < self.angle_dead_zone:
            return MovementCommand(forward=True, turn=0)

        # 误差很大：先纯转向，不前进
        if abs(diff) > self.angle_slow_turn:
            turn_dir = -1 if diff < 0 else 1
            return MovementCommand(forward=False, turn=turn_dir)

        # 误差中等：一边小幅转向一边前进
        turn_dir = -1 if diff < 0 else 1
        return MovementCommand(forward=True, turn=turn_dir)
