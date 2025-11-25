from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

# 按你的工程结构修改导入路径
from src.attack.aim_controller import AimCommand


@dataclass
class FireDecision:
    """射击决策输出。

    should_fire: 本帧是否需要触发一次左键点击（短按）。
    reason: 触发原因，便于日志与调试。
    """

    should_fire: bool
    reason: str = ""


class FireController:
    """基于瞄准稳定性 + 简单装填时间的射击控制器。

    当前版本不做复杂的“缩圈识别”，而是用：
    - 目标存在且连续稳定瞄准至少 min_stable_time 秒
    - 距离上一次开火至少 reload_time 秒
    即可判定为可以开火。

    后续可以在此基础上接入真正的缩圈检测模块，替换“稳定时间”这一近似条件。
    """

    def __init__(
        self,
        min_stable_time: float = 0.8,
        reload_time: float = 8.0,
        require_target_on_fire: bool = True,
    ) -> None:
        """初始化 FireController。

        Args:
            min_stable_time: 连续保持 AimCommand.is_stable=True 的最短时间（秒），视作瞄准完成
            reload_time: 简化版装填时间（秒），两次开火之间的最小间隔
            require_target_on_fire: 开火时是否必须存在目标（一般为 True）
        """

        if min_stable_time <= 0:
            raise ValueError("min_stable_time 必须为正数")
        if reload_time <= 0:
            raise ValueError("reload_time 必须为正数")

        self.min_stable_time = float(min_stable_time)
        self.reload_time = float(reload_time)
        self.require_target_on_fire = bool(require_target_on_fire)

        # 内部状态
        self._stable_time_acc: float = 0.0  # 当前连续稳定瞄准的累积时间
        self._time_since_last_shot: float = 1e9  # 距离上一次开火的时间

    # ------------------------------------------------------------------ #
    # 对外主接口
    # ------------------------------------------------------------------ #
    def update(
        self,
        dt: float,
        has_target: bool,
        aim_cmd: AimCommand,
    ) -> FireDecision:
        """每帧更新射击控制逻辑。

        Args:
            dt: 本帧时间间隔（秒），由控制线程传入
            has_target: 当前是否存在主攻击目标
            aim_cmd: 本帧瞄准控制器的输出

        Returns:
            FireDecision: 本帧是否需要触发一次开火
        """

        # 防御式：dt <= 0 时不更新内部时间，直接认为不能开火
        if dt <= 0:
            logger.warning("FireController.update: dt <= 0, 跳过本帧更新时间")
            return FireDecision(should_fire=False, reason="invalid_dt")

        # 时间推进
        self._time_since_last_shot += dt

        # 1) 更新稳定瞄准时间
        if has_target and aim_cmd.is_stable:
            self._stable_time_acc += dt
        else:
            # 目标丢失或尚不稳定时，清零稳定时间
            self._stable_time_acc = 0.0

        # 2) 各条件检查
        if self.require_target_on_fire and not has_target:
            return FireDecision(should_fire=False, reason="no_target")

        if not aim_cmd.is_stable:
            return FireDecision(should_fire=False, reason="aim_not_stable")

        if self._stable_time_acc < self.min_stable_time:
            return FireDecision(should_fire=False, reason="stable_time_not_enough")

        if self._time_since_last_shot < self.reload_time:
            return FireDecision(should_fire=False, reason="reloading")

        # 3) 所有条件满足，触发一次开火
        self._time_since_last_shot = 0.0
        self._stable_time_acc = 0.0  # 重新开始累计稳定时间

        logger.debug("FireController: 开火条件满足，发射！")
        return FireDecision(should_fire=True, reason="fire")


# ---------------------------------------------------------------------- #
# Demo: 简单时间序列测试 FireController
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    # 模拟 0.1 秒一次的控制循环
    dt = 0.1
    fire_ctrl = FireController(min_stable_time=0.8, reload_time=3.0)

    # 连续 50 帧，前几帧不稳定，后面稳定
    timeline = []
    for i in range(50):
        # 假设从第 5 帧开始有目标，从第 10 帧开始 is_stable=True
        has_target = i >= 5
        is_stable = i >= 10
        aim_cmd = AimCommand(dx=0, dy=0, is_stable=is_stable)

        decision = fire_ctrl.update(dt=dt, has_target=has_target, aim_cmd=aim_cmd)
        timeline.append((i, has_target, is_stable, decision))

    for i, has_target, is_stable, decision in timeline:
        logger.info(
            f"frame={i:02d}, has_target={has_target}, stable={is_stable}, "
            f"should_fire={decision.should_fire}, reason={decision.reason}"
        )
