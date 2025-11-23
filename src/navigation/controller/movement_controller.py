from __future__ import annotations

import math
from loguru import logger
from src.navigation.controller.move_executor import MovementCommand
from src.navigation.controller.move_executor import MoveExecutor

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

        self.angle_dead_zone = math.radians(angle_dead_zone_deg)
        self.angle_slow_turn = math.radians(angle_slow_turn_deg)
        self.dist_stop = distance_stop_threshold
        self.dist_slow = slow_down_distance
        self.max_forward_speed = max(0.0, max_forward_speed)
        self.min_forward_factor = max(0.0, min(1.0, min_forward_factor))
        self.large_angle_threshold = math.radians(large_angle_threshold_deg)
        self.large_angle_speed_reduction = max(0.0, min(1.0, large_angle_speed_reduction))

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
        #   - 主要基于距离因子
        #   - 角度误差影响较小，仅在极大角度时进行额外衰减
        # 距离因子
        if dist >= self.dist_slow:
            dist_factor = 1.0
        else:
            # 在 [dist_stop, dist_slow] 间线性插值
            dist_factor = (dist - self.dist_stop) / max(1e-3, self.dist_slow - self.dist_stop)
            dist_factor = max(0.0, min(1.0, dist_factor))

        # 角度因子：不再直接清零，而是线性衰减到最小值
        angle_factor = 1.0 - abs_diff / self.angle_slow_turn
        angle_factor = max(self.min_forward_factor, angle_factor)
        angle_factor = min(1.0, angle_factor)

        # 基础前进速度主要基于距离
        forward_cmd = self.max_forward_speed * dist_factor

        # 仅在极大角度误差时进行额外的乘性衰减
        if abs_diff > self.large_angle_threshold:
            forward_cmd *= self.large_angle_speed_reduction

        # 应用角度因子（但不会完全停止）
        forward_cmd *= angle_factor

        # 调试日志
        logger.debug(
            f"decide: dist={dist:.1f}, angle_err={math.degrees(abs_diff):.1f}deg | "
            f"dist_factor={dist_factor:.3f}, angle_factor={angle_factor:.3f} | "
            f"forward_cmd={forward_cmd:.3f}, turn_cmd={turn_cmd:.3f}"
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