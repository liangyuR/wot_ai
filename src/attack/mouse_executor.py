from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from loguru import logger

try:
    import ctypes
    from ctypes import wintypes
except ImportError:  # 非 Windows 环境下导入会失败
    ctypes = None  # type: ignore[assignment]
    wintypes = None  # type: ignore[assignment]


class MouseExecutor(Protocol):
    """鼠标执行器抽象接口。

    AttackModule 不直接依赖具体实现，只依赖这个接口：
    - move_relative: 按像素相对移动鼠标
    - click_left:    执行一次左键点击（短按）
    """

    def move_relative(self, dx: int, dy: int) -> None:  # pragma: no cover - interface
        ...

    def click_left(self) -> None:  # pragma: no cover - interface
        ...


@dataclass
class Win32MouseExecutor:
    """基于 Win32 API 的鼠标执行器实现（仅支持 Windows）。

    使用 mouse_event 发送相对移动与左键按下/抬起。
    """

    move_scale: float = 1.0  # 鼠标移动缩放系数（方便后续整体调灵敏度）

    def __post_init__(self) -> None:
        if ctypes is None:
            raise RuntimeError("Win32MouseExecutor 只能在 Windows 环境下使用（ctypes 导入失败）")

        self.user32 = ctypes.WinDLL("user32", use_last_error=True)

        # mouse_event 标志位
        self.MOUSEEVENTF_MOVE = 0x0001
        self.MOUSEEVENTF_LEFTDOWN = 0x0002
        self.MOUSEEVENTF_LEFTUP = 0x0004

    # ------------------------------------------------------------------ #
    # MouseExecutor 接口实现
    # ------------------------------------------------------------------ #
    def move_relative(self, dx: int, dy: int) -> None:
        """相对移动鼠标光标（屏幕坐标系下 dx, dy 像素）。

        注意：dx, dy 会乘以 move_scale 后取整。
        """
        if dx == 0 and dy == 0:
            return

        sx = int(round(dx * self.move_scale))
        sy = int(round(dy * self.move_scale))

        # 使用 mouse_event 发送相对移动
        self.user32.mouse_event(
            self.MOUSEEVENTF_MOVE,
            sx,
            sy,
            0,
            0,
        )

    def click_left(self) -> None:
        """执行一次左键点击（按下 + 抬起）。"""
        # 左键按下
        self.user32.mouse_event(
            self.MOUSEEVENTF_LEFTDOWN,
            0,
            0,
            0,
            0,
        )
        # 左键抬起
        self.user32.mouse_event(
            self.MOUSEEVENTF_LEFTUP,
            0,
            0,
            0,
            0,
        )


# ---------------------------------------------------------------------- #
# Demo: 简单测试 Win32MouseExecutor
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import time

    logger.info("Win32MouseExecutor demo: 3 秒后开始轻微移动鼠标并点击一次左键，请切到安全窗口。")
    time.sleep(3.0)

    mouse = Win32MouseExecutor(move_scale=1.0)

    logger.info("尝试向右下角移动 50 像素 ...")
    mouse.move_relative(50, 50)

    time.sleep(0.5)
    logger.info("尝试点击一次左键 ...")
    mouse.click_left()

    logger.info("demo 结束")
