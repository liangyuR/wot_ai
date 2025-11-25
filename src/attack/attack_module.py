from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Protocol

import numpy as np
from loguru import logger

# 按你的工程结构修改导入路径
from src.attack.main_view_detector import (
    MainViewDetector,
    ScreenTarget,
)
from src.attack.target_selector import AttackTargetSelector
from src.attack.aim_controller import AimController, AimCommand
from src.attack.fire_controller import FireController, FireDecision
from src.attack.mouse_executor import MouseExecutor


@dataclass
class AttackStats:
    """简单统计信息，方便调试和可视化。"""

    last_has_target: bool = False
    last_fired: bool = False
    last_fire_reason: str = ""
    last_target_center: Optional[Tuple[int, int]] = None


class AttackModule:
    """攻击模块高层封装：检测 → 选目标 → 瞄准 → 开火。

    本模块本身 **不管理线程**，由上层 Runtime 在固定 tick 中调用
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
            return

        h, w = frame_bgr.shape[:2]
        screen_center = (w // 2, h // 2)

        # 1) 视觉检测：主视野敌人列表
        targets = self.detector.detect(frame_bgr)

        # 2) 目标选择
        best: Optional[ScreenTarget] = self.selector.update(targets, screen_center)
        has_target = best is not None

        # 3) 瞄准控制：得到本帧鼠标移动命令
        aim_cmd: AimCommand = self.aim.update(best)

        # 4) 执行鼠标移动
        if aim_cmd.dx or aim_cmd.dy:
            try:
                self.mouse.move_relative(aim_cmd.dx, aim_cmd.dy)
            except Exception as e:  # 保底，避免因为鼠标错误导致线程挂掉
                logger.exception(f"AttackModule: mouse.move_relative 失败: {e}")

        # 5) 射击决策
        fire_decision: FireDecision = self.fire.update(
            dt=dt,
            has_target=has_target,
            aim_cmd=aim_cmd,
        )

        # 6) 执行开火
        fired = False
        if fire_decision.should_fire:
            try:
                self.mouse.click_left()
                fired = True
            except Exception as e:
                logger.exception(f"AttackModule: mouse.click_left 失败: {e}")

        # 7) 更新统计信息（方便 debug overlay 使用）
        self.stats.last_has_target = has_target
        self.stats.last_fired = fired
        self.stats.last_fire_reason = fire_decision.reason
        self.stats.last_target_center = best.center if best is not None else None

        logger.debug(
            "AttackModule.update: has_target=%s, fired=%s, reason=%s, target_center=%s",
            has_target,
            fired,
            fire_decision.reason,
            self.stats.last_target_center,
        )


# ---------------------------------------------------------------------- #
# Demo: 结构完整性检查（不做真实检测/鼠标操作）
# ---------------------------------------------------------------------- #

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
