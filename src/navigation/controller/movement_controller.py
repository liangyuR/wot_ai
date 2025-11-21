from __future__ import annotations

import math
from src.navigation.controller.move_executor import MovementCommand

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
    ) -> None:
        """初始化控制参数。

        Args:
            angle_dead_zone_deg: 朝向误差小于该角度时认为对准，转向输出趋近于 0
            angle_slow_turn_deg: 误差大于该角度时优先转向，前进速度线性衰减至 0
            distance_stop_threshold: 距离目标小于该值时停止前进
            slow_down_distance: 开始减速的距离（大于 stop_threshold）
            max_forward_speed: 最大前进速度（映射到 MovementCommand.forward，通常在 [0, 1]）
        """
        if slow_down_distance < distance_stop_threshold:
            slow_down_distance = distance_stop_threshold

        self.angle_dead_zone = math.radians(angle_dead_zone_deg)
        self.angle_slow_turn = math.radians(angle_slow_turn_deg)
        self.dist_stop = distance_stop_threshold
        self.dist_slow = slow_down_distance
        self.max_forward_speed = max(0.0, max_forward_speed)

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
        current_pos: tuple[float, float],
        heading: float,
        target_pos: tuple[float, float],
    ) -> MovementCommand:
        """核心决策：输出连续的前进/转向控制量。

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
        if dist < self.dist_stop or self.max_forward_speed <= 0.0:
            return MovementCommand(forward=0.0, turn=0.0, brake=False)

        # 目标方向角
        desired = math.atan2(dy, dx)
        diff = self._norm_angle(desired - heading)
        abs_diff = abs(diff)

        # 1) 计算转向控制量：按角度误差线性归一到 [-1, 1]
        if abs_diff < self.angle_dead_zone:
            turn_cmd = 0.0
        else:
            # 将误差缩放到 [-1, 1] 区间之外，然后截断
            turn_cmd = diff / self.angle_slow_turn
            turn_cmd = max(-1.0, min(1.0, turn_cmd))

        # 2) 计算前进速度：
        #   - 距离越近越慢
        #   - 角度误差越大越慢，超出 angle_slow_turn 时逐渐接近 0
        # 距离因子
        if dist >= self.dist_slow:
            dist_factor = 1.0
        else:
            # 在 [dist_stop, dist_slow] 间线性插值
            dist_factor = (dist - self.dist_stop) / max(1e-3, self.dist_slow - self.dist_stop)
            dist_factor = max(0.0, min(1.0, dist_factor))

        # 角度因子
        if abs_diff >= self.angle_slow_turn:
            angle_factor = 0.0
        else:
            angle_factor = 1.0 - abs_diff / self.angle_slow_turn
            angle_factor = max(0.0, min(1.0, angle_factor))

        forward_cmd = self.max_forward_speed * dist_factor * angle_factor

        return MovementCommand(forward=forward_cmd, turn=turn_cmd, brake=False)
