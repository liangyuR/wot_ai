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
    """基于瞄准圈半径 + 瞄准稳定性 + 装填时间的射击控制器。

    核心逻辑（看圈开火版本）：

    - 必须有攻击目标（has_target=True）
    - AimController 报告 is_stable=True（目标已基本居中）
    - 检测到瞄准圈（reticle_radius 不为 None）
    - 瞄准圈半径小于阈值 reticle_radius_fire_threshold，且持续至少
      min_reticle_stable_time 秒
    - 距离上一次开火至少 reload_time 秒

    为了兼容旧版本，还保留了“只看稳定时间”的退化逻辑（allow_fire_without_reticle）。
    """

    def __init__(
        self,
        # 老参数：依然保留
        min_stable_time: float = 0.8,
        reload_time: float = 8.0,
        require_target_on_fire: bool = True,
        # 新参数：看圈相关
        use_reticle: bool = True,
        reticle_radius_fire_threshold: float = 25.0,
        min_reticle_stable_time: float = 0.3,
        allow_fire_without_reticle: bool = False,
    ) -> None:
        """初始化 FireController。

        Args:
            min_stable_time:
                仅使用“稳定时间”逻辑时，连续保持 AimCommand.is_stable=True
                的最短时间（秒），视作瞄准完成。
                当 use_reticle=True 且 reticle 可用时，这个条件只作为兜底。

            reload_time:
                简化版装填时间（秒），两次开火之间的最小间隔。

            require_target_on_fire:
                开火时是否必须存在目标（一般为 True）。

            use_reticle:
                是否启用“看圈开火”逻辑。若为 False，则退化为只看稳定时间。

            reticle_radius_fire_threshold:
                认为“圈已经缩好”的半径阈值（像素）。
                瞄准圈半径 <= 该值，且持续 min_reticle_stable_time 秒，才允许开火。

            min_reticle_stable_time:
                瞄准圈半径连续满足阈值条件的最短时间（秒），用于防抖。

            allow_fire_without_reticle:
                当 use_reticle=True 但当前帧没有 reticle_radius 时，
                是否允许退化为仅按稳定时间逻辑开火。
        """

        if min_stable_time <= 0:
            raise ValueError("min_stable_time 必须为正数")
        if reload_time <= 0:
            raise ValueError("reload_time 必须为正数")
        if reticle_radius_fire_threshold <= 0:
            raise ValueError("reticle_radius_fire_threshold 必须为正数")
        if min_reticle_stable_time <= 0:
            raise ValueError("min_reticle_stable_time 必须为正数")

        self.min_stable_time = float(min_stable_time)
        self.require_target_on_fire = bool(require_target_on_fire)

        self.use_reticle = bool(use_reticle)
        self.reticle_radius_fire_threshold = float(reticle_radius_fire_threshold)
        self.min_reticle_stable_time = float(min_reticle_stable_time)
        self.allow_fire_without_reticle = bool(allow_fire_without_reticle)

        # 内部状态
        self._stable_time_acc: float = 0.0           # 准星稳定时间累积
        self._reticle_stable_time_acc: float = 0.0   # 瞄准圈满足阈值的累积时间
        self._time_since_last_shot: float = 1e9      # 距离上一次开火的时间

    # ------------------------------------------------------------------ #
    # 对外主接口
    # ------------------------------------------------------------------ #
    def update(
        self,
        dt: float,
        has_target: bool,
        aim_cmd: AimCommand,
        reticle_radius: Optional[float] = None,
    ) -> FireDecision:
        """每帧更新射击控制逻辑。

        Args:
            dt: 本帧时间间隔（秒），由控制线程传入
            has_target: 当前是否存在主攻击目标（真实目标，不含预瞄）
            aim_cmd: 本帧瞄准控制器的输出
            reticle_radius: 当前帧瞄准圈半径（像素），若未检测到则为 None

        Returns:
            FireDecision: 本帧是否需要触发一次开火
        """

        # 防御式：dt <= 0 时不更新内部时间，直接认为不能开火
        if dt <= 0:
            logger.warning("FireController.update: dt <= 0, 跳过本帧更新时间")
            return FireDecision(should_fire=False, reason="invalid_dt")

        # 时间推进
        self._time_since_last_shot += dt

        # 1) 更新“准星稳定时间”
        if has_target and aim_cmd.is_stable:
            self._stable_time_acc += dt
        else:
            self._stable_time_acc = 0.0

        # 2) 更新“瞄准圈满足阈值的稳定时间”
        if (
            self.use_reticle
            and has_target
            and aim_cmd.is_stable
            and reticle_radius is not None
            and reticle_radius <= self.reticle_radius_fire_threshold
        ):
            self._reticle_stable_time_acc += dt
        else:
            # 只要条件不满足就清零，避免短暂满足立刻开火
            self._reticle_stable_time_acc = 0.0

        # 3) 基础条件检查
        if self.require_target_on_fire and not has_target:
            return FireDecision(should_fire=False, reason="no_target")

        if not aim_cmd.is_stable:
            return FireDecision(should_fire=False, reason="aim_not_stable")

        # 5) 看圈开火逻辑
        if self.use_reticle:
            if reticle_radius is None:
                # 没有圈：根据配置决定是否退化为“只看稳定时间”
                if not self.allow_fire_without_reticle:
                    return FireDecision(should_fire=False, reason="no_reticle")
                # 允许退化，则走旧逻辑
            else:
                # 有圈：必须半径足够小，且稳定时间足够
                if reticle_radius > self.reticle_radius_fire_threshold:
                    return FireDecision(should_fire=False, reason="reticle_not_shrunk")

                if self._reticle_stable_time_acc < self.min_reticle_stable_time:
                    return FireDecision(
                        should_fire=False,
                        reason="reticle_stable_time_not_enough",
                    )

        # 7) 所有条件满足，触发一次开火
        self._time_since_last_shot = 0.0
        self._stable_time_acc = 0.0
        self._reticle_stable_time_acc = 0.0

        logger.debug("FireController: 开火条件满足，发射！")
        return FireDecision(should_fire=True, reason="fire")


# ---------------------------------------------------------------------- #
# Demo: 简单时间序列测试 FireController（含看圈逻辑）
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    # 模拟 0.1 秒一次的控制循环
    dt = 0.1
    fire_ctrl = FireController(
        min_stable_time=0.5,
        reload_time=3.0,
        use_reticle=True,
        reticle_radius_fire_threshold=25.0,
        min_reticle_stable_time=0.3,
        allow_fire_without_reticle=False,
    )

    # 连续 50 帧，模拟：从第 5 帧开始有目标，从第 10 帧开始 is_stable=True，
    # 瞄准圈半径线性缩小，直到小于阈值。
    timeline = []
    radius = 60.0
    for i in range(50):
        has_target = i >= 5
        is_stable = i >= 10

        # 模拟圈缩小
        if i >= 10:
            radius = max(10.0, radius - 2.0)
            reticle_radius = radius
        else:
            reticle_radius = None

        aim_cmd = AimCommand(dx=0, dy=0, is_stable=is_stable)

        decision = fire_ctrl.update(
            dt=dt,
            has_target=has_target,
            aim_cmd=aim_cmd,
            reticle_radius=reticle_radius,
        )
        timeline.append((i, has_target, is_stable, reticle_radius, decision))

    for i, has_target, is_stable, r, decision in timeline:
        logger.info(
            f"frame={i:02d}, has_target={has_target}, stable={is_stable}, "
            f"reticle_radius={r}, should_fire={decision.should_fire}, "
            f"reason={decision.reason}"
        )
