from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

import numpy as np
from loguru import logger

# 按你的工程结构修改导入路径
from src.attack.main_view_detector import (
    MainViewDetector,
    ScreenTarget,
    AimReticle,   
    HpBarDet,     # 预瞄用
)
from src.attack.target_selector import AttackTargetSelector
from src.attack.aim_controller import AimController, AimCommand
from src.attack.fire_controller import FireController, FireDecision
from src.attack.mouse_executor import MouseExecutor


@dataclass
class AttackStats:
    """简单统计信息，方便调试和可视化。"""

    last_has_target: bool = False          # 是否有真实攻击目标
    last_fired: bool = False
    last_fire_reason: str = ""
    last_target_center: Optional[Tuple[int, int]] = None   # 本帧瞄准的中心（真实 or 预瞄）
    last_is_pre_aim: bool = False          # 本帧是否处于“预瞄”状态（仅血条）


class AttackModule:
    """攻击模块高层封装：检测 → 选目标 → 预瞄 → 瞄准 → 开火。

    本模块本身不管理线程，由上层 Runtime 在固定 tick 中调用
    `update(frame_bgr, dt)` 即可。
    """

    def __init__(self) -> None:
        self.detector = MainViewDetector()
        self.selector = AttackTargetSelector()
        self.aim = AimController()
        self.fire = FireController()
        self.mouse = MouseExecutor()

        self.stats = AttackStats()

    # ------------------------------------------------------------------ #
    # 对外主接口
    # ------------------------------------------------------------------ #
    def update(self, frame_bgr: np.ndarray, dt: float) -> None:
        """执行一次完整攻击循环。

        Args:
            frame_bgr: 当前帧主视野图像（BGR 格式，整屏或主视野区域）
            dt: 上一帧到当前帧的时间间隔（秒）
        """

        if frame_bgr is None or frame_bgr.size == 0:
            logger.debug("AttackModule.update: frame 为空，跳过本帧")
            self.stats.last_has_target = False
            self.stats.last_fired = False
            self.stats.last_fire_reason = "empty_frame"
            self.stats.last_target_center = None
            self.stats.last_is_pre_aim = False
            return

        h, w = frame_bgr.shape[:2]
        screen_center = (w // 2, h // 2)

        # 1) 视觉检测：主视野敌人列表 + 瞄准圈 + 敌方血条
        targets, reticle, hp_bars = self.detector.detect(frame_bgr)

        # 2) 目标选择（真实目标）
        best: Optional[ScreenTarget] = None
        if targets:
            best = self.selector.update(targets, screen_center)

        has_real_target = best is not None

        # 3) 预瞄逻辑：若没有真实目标，但存在敌方血条，则构造预瞄目标
        pre_aim_target: Optional[ScreenTarget] = None
        if (best is None) and hp_bars:
            hp = self._choose_hpbar_for_pre_aim(hp_bars, screen_center)
            if hp is not None:
                pre_aim_target = self._build_pre_aim_target(hp, frame_h=h)

        # 4) 瞄准控制：优先瞄准真实目标，否则瞄准预瞄点
        aim_target: Optional[ScreenTarget] = best if best is not None else pre_aim_target
        aim_cmd: AimCommand = self.aim.update(aim_target)

        # 5) 执行鼠标移动
        if aim_cmd.dx or aim_cmd.dy:
            try:
                self.mouse.move_relative(aim_cmd.dx, aim_cmd.dy)
            except Exception as e:  # 保底，避免因为鼠标错误导致线程挂掉
                logger.exception(f"AttackModule: mouse.move_relative 失败: {e}")

        # 6) 射击决策：只认“真实目标”存在的情况下才考虑开火
        fire_decision: FireDecision = self.fire.update(
            dt=dt,
            has_target=has_real_target,
            aim_cmd=aim_cmd,
        )

        # 7) 执行开火
        fired = False
        if fire_decision.should_fire:
            try:
                self.mouse.click_left()
                fired = True
            except Exception as e:
                logger.exception(f"AttackModule: mouse.click_left 失败: {e}")

        # 8) 更新统计信息（方便 debug overlay 使用）
        is_pre_aim = (best is None) and (pre_aim_target is not None)
        aim_center = None
        if aim_target is not None:
            aim_center = aim_target.center

        self.stats.last_has_target = has_real_target
        self.stats.last_fired = fired
        self.stats.last_fire_reason = fire_decision.reason
        self.stats.last_target_center = aim_center
        self.stats.last_is_pre_aim = is_pre_aim

        logger.debug(
            "AttackModule.update: has_real_target=%s, pre_aim=%s, fired=%s, "
            "reason=%s, target_center=%s",
            has_real_target,
            is_pre_aim,
            fired,
            fire_decision.reason,
            self.stats.last_target_center,
        )

    # ------------------------------------------------------------------ #
    # 预瞄相关内部方法
    # ------------------------------------------------------------------ #
    def _choose_hpbar_for_pre_aim(
        self,
        hp_bars: Sequence[HpBarDet],
        screen_center: Tuple[int, int],
    ) -> Optional[HpBarDet]:
        """从多个血条中选择一个用于预瞄（当前策略：离屏幕中心最近）。"""
        if not hp_bars:
            return None

        sx, sy = screen_center
        best_hp: Optional[HpBarDet] = None
        best_d2: float = 1e18

        for hp in hp_bars:
            x1, y1, x2, y2 = hp.bbox
            hcx = (x1 + x2) // 2
            hcy = (y1 + y2) // 2
            dx = hcx - sx
            dy = hcy - sy
            d2 = float(dx * dx + dy * dy)
            if d2 < best_d2:
                best_d2 = d2
                best_hp = hp

        return best_hp

    def _build_pre_aim_target(self, hp: HpBarDet, frame_h: int) -> ScreenTarget:
        """根据血条位置构造一个“虚拟目标”，用于预瞄。

        策略（简化版）：
        - 取血条中心点 (hcx, hcy)
        - 向下偏移 base_offset + k * hp_height 得到预瞄点
        - 以预瞄点为中心构造一个小 bbox，给 AimController 用
        """
        x1, y1, x2, y2 = hp.bbox
        hcx = (x1 + x2) // 2
        hcy = (y1 + y2) // 2
        h_height = max(1, y2 - y1)

        base_offset = 20       # 固定向下偏移，可以挪到配置
        k = 1.2                # 跟血条高度的缩放系数，后续可以用数据拟合优化
        offset_y = base_offset + int(k * h_height)

        py = hcy + offset_y
        py = min(max(py, 0), frame_h - 1)

        center = (hcx, py)

        # 用一个很小的 bbox 包围预瞄点，主要是为了可视化 / 调试
        half_box = 5
        bx1 = hcx - half_box
        by1 = py - half_box
        bx2 = hcx + half_box
        by2 = py + half_box

        return ScreenTarget(
            bbox=(bx1, by1, bx2, by2),
            center=center,
            score=float(hp.score),
            cls_id=0,          # 这里随便给一个类别；如需区分可后续增加字段
            is_enemy=True,
            est_distance=None,
        )


# ---------------------------------------------------------------------- #
# Demo: 真实攻击测试 - 启动线程，抓取窗口并攻击
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import threading
    import time

    from src.navigation.service.capture_service import CaptureService
    from src.attack.main_view_detector import MainViewDetector
    from src.attack.target_selector import AttackTargetSelector
    from src.attack.aim_controller import AimController
    from src.attack.fire_controller import FireController

    # 1. 初始化截图服务，绑定到 WOT 窗口
    capture_service = CaptureService()
    ok = capture_service.grab_window_by_name("WorldOfTanks")
    if not ok:
        logger.error("无法绑定 WorldOfTanks 窗口，退出")
        raise SystemExit(1)

    # 2. 抓一帧确定分辨率
    first_frame = capture_service.grab()
    if first_frame is None:
        logger.error("首次抓取画面失败，退出")
        raise SystemExit(1)

    h, w = first_frame.shape[:2]
    screen_center = (w // 2, h // 2)
    logger.info(f"攻击 Demo: 绑定窗口分辨率 {w}x{h}, screen_center={screen_center}")

    # 3. 构造真实模块
    module = AttackModule()

    # 4. 攻击线程函数
    def attack_loop() -> None:
        logger.info("Attack thread started")

        target_fps = 15
        interval = 1.0 / target_fps
        last_time = time.perf_counter()

        while True:
            t0 = time.perf_counter()
            dt = t0 - last_time
            last_time = t0

            try:
                frame = capture_service.grab()
                if frame is None:
                    time.sleep(0.01)
                    continue

                module.update(frame_bgr=frame, dt=dt)

            except Exception as e:
                logger.exception(f"Attack loop error: {e}")

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    # 5. 启动线程并保持主线程阻塞，按 Ctrl+C 退出
    t = threading.Thread(target=attack_loop, daemon=True)
    t.start()

    logger.info("Attack demo 正在运行，按 Ctrl+C 退出 ...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，退出攻击 demo")
