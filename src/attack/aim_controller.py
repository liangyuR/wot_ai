from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from loguru import logger

# 按你的工程结构修改导入路径
from src.attack.main_view_detector import ScreenTarget


@dataclass
class AimCommand:
    """瞄准控制输出：用于驱动鼠标相对移动。

    dx, dy 为本帧期望的鼠标相对位移（像素），
    is_stable 表示当前准心是否已经基本稳定在目标附近。
    """

    dx: int
    dy: int
    is_stable: bool


class AimController:
    """将目标屏幕坐标转换为鼠标相对移动命令的控制器。

    职责：
    - 接收当前主目标 (ScreenTarget) 和屏幕中心坐标
    - 根据两者的偏差，计算本帧需要的鼠标 dx, dy
    - 提供简单的死区与最大步长限制，让瞄准动作更加平滑
    """

    def __init__(
        self,
        screen_center: Tuple[int, int],
        kp: float = 0.18,
        max_step_px: int = 30,
        dead_zone_px: int = 5,
    ) -> None:
        """初始化 AimController。

        Args:
            screen_center: 屏幕中心像素坐标 (cx, cy)
            kp: 比例系数，将目标偏差转换为鼠标移动量
            max_step_px: 单帧最大鼠标移动像素（防止甩枪）
            dead_zone_px: 死区半径，偏差小于该值时认为已经对准
        """

        self.screen_center = screen_center
        self.kp = float(kp)
        self.max_step_px = max(1, int(max_step_px))
        self.dead_zone_px = max(1, int(dead_zone_px))

    # ------------------------------------------------------------------ #
    # 对外主接口
    # ------------------------------------------------------------------ #
    def update(self, target: Optional[ScreenTarget]) -> AimCommand:
        """根据当前主目标更新瞄准命令。

        Args:
            target: 主攻击目标，若为 None 则不移动鼠标。

        Returns:
            AimCommand: 本帧期望的鼠标相对移动命令。
        """

        if target is None:
            # 没有目标时保持不动，认为不稳定（方便 FireController 判断）
            return AimCommand(dx=0, dy=0, is_stable=False)

        cx, cy = self.screen_center
        tx, ty = target.center

        # 目标相对于屏幕中心的误差（像素）
        ex = tx - cx
        ey = ty - cy

        # 偏差模长
        dist = (ex * ex + ey * ey) ** 0.5

        # 死区判断：在一定范围内认为已经对准
        if dist <= self.dead_zone_px:
            return AimCommand(dx=0, dy=0, is_stable=True)

        # 比例控制：将误差缩放为鼠标位移
        # 注意：鼠标向右为正 x，向下为正 y，与屏幕坐标一致
        raw_dx = self.kp * ex
        raw_dy = self.kp * ey

        # 限制单帧最大移动，避免甩枪
        dx, dy = self._limit_step(raw_dx, raw_dy)

        # 只要还在移动，就认为尚未完全稳定
        return AimCommand(dx=dx, dy=dy, is_stable=False)

    # ------------------------------------------------------------------ #
    # 内部工具函数
    # ------------------------------------------------------------------ #
    def _limit_step(self, dx: float, dy: float) -> Tuple[int, int]:
        """对 (dx, dy) 进行幅度裁剪，限制单帧最大步长。"""
        import math

        mag = math.hypot(dx, dy)
        if mag <= 1e-6:
            return 0, 0

        if mag <= self.max_step_px:
            return int(round(dx)), int(round(dy))

        scale = self.max_step_px / mag
        return int(round(dx * scale)), int(round(dy * scale))


# ---------------------------------------------------------------------- #
# Demo: 用假数据测试 AimController 输出
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    # 假设 1280x720，屏幕中心：
    screen_center = (1280 // 2, 720 // 2)
    aim = AimController(screen_center=screen_center, kp=0.18, max_step_px=30, dead_zone_px=5)

    # 造几个假目标中心测试
    fake_targets = [
        ScreenTarget(bbox=(600, 300, 650, 350), center=(625, 325), score=0.9, cls_id=0, is_enemy=True),
        ScreenTarget(bbox=(900, 400, 950, 450), center=(925, 425), score=0.9, cls_id=0, is_enemy=True),
        ScreenTarget(bbox=(screen_center[0] + 3, screen_center[1] + 2,
                           screen_center[0] + 10, screen_center[1] + 9),
                     center=(screen_center[0] + 3, screen_center[1] + 2),
                     score=0.9, cls_id=0, is_enemy=True),
    ]

    for i, t in enumerate(fake_targets, 1):
        cmd = aim.update(t)
        logger.info(f"case#{i}: target_center={t.center}, aim_cmd={cmd}")

    # 测试无目标时输出
    cmd_none = aim.update(None)
    logger.info(f"no target: aim_cmd={cmd_none}")
