#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-performance screen capture service with buffer reuse.

- Focuses on full-screen and region capture only (no overlay logic)
- Thread-safe via thread-local MSS instances
- Reuses per-thread numpy buffers to minimize allocations
"""

from __future__ import annotations

from typing import Optional, Tuple
import threading

import cv2  # noqa: F401  # kept for potential future use
import numpy as np
from loguru import logger


class CaptureService:
    """Screen capture service for full-screen and arbitrary regions.

    Responsibilities:
    - Initialize screen information (width, height) for a given monitor
    - Provide fast BGR frames for full-screen or region capture
    - Reuse buffers per thread to reduce memory allocations
    """

    def __init__(self, monitor_index: int = 1) -> None:
        """Create a capture service bound to a specific monitor.

        Args:
            monitor_index: MSS monitor index.
                0: all monitors bounding box
                1: primary monitor
                2+: additional monitors
        """
        self._monitor_index = monitor_index
        self._valid = False
        self._tls = threading.local()  # per-thread context (sct + buffer)
        self.pid = find_pid_by_process_name()
        if self.pid is None:
            raise Exception("WorldOfTanks 进程未找到")
        self.window_rect = find_window_rect_by_pid(self.pid)
        if self.window_rect is None:
            raise Exception("WorldOfTanks 窗口未找到")
        self.window_left = self.window_rect[0]
        self.window_top = self.window_rect[1]
        self.window_width = self.window_rect[2] - self.window_rect[0]
        self.window_height = self.window_rect[3] - self.window_rect[1]

        try:
            import mss  # type: ignore

            self._mss_module = mss
            temp_sct = mss.mss()

            monitors = temp_sct.monitors
            if monitor_index < 0 or monitor_index >= len(monitors):
                raise IndexError(f"Invalid monitor index: {monitor_index}, available: 0..{len(monitors) - 1}")

            self._monitor = monitors[monitor_index]
            self.width: int = int(self._monitor["width"])
            self.height: int = int(self._monitor["height"])

            self._valid = True
            logger.info(f"CaptureService initialized: monitor={monitor_index}, size={self.width}x{self.height}")
        except ImportError:
            logger.error("mss is not installed. Please run: pip install mss")
        except Exception as e:  # noqa: BLE001
            logger.error(f"CaptureService initialization failed: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_thread_ctx(self) -> None:
        """Ensure current thread has its own MSS instance and buffer slot."""
        if not hasattr(self._tls, "ctx"):
            # Each thread has its own MSS instance and buffer
            sct = self._mss_module.mss() if getattr(self, "_valid", False) else None
            self._tls.ctx = {"sct": sct, "buffer": None}

    def _get_thread_ctx(self):
        self._ensure_thread_ctx()
        return self._tls.ctx

    def _capture_to_buffer(self, mon: dict) -> Optional[np.ndarray]:
        """Capture a region described by `mon` into a reusable BGR buffer.

        Args:
            mon: MSS region dict with keys: left, top, width, height.

        Returns:
            A numpy.ndarray (H, W, 3) in BGR format, or None on failure.
        """
        if not self._valid:
            return None

        ctx = self._get_thread_ctx()
        sct = ctx["sct"]
        if sct is None:
            return None

        try:
            shot = sct.grab(mon)  # BGRA
            bgra = np.asarray(shot)

            h, w = bgra.shape[:2]
            buf: Optional[np.ndarray] = ctx["buffer"]

            # Allocate once per thread / resolution
            if buf is None or buf.shape[0] != h or buf.shape[1] != w:
                buf = np.empty((h, w, 3), dtype=np.uint8)
                ctx["buffer"] = buf

            # Drop alpha channel, keep BGR
            buf[...] = bgra[:, :, :3]
            return buf
        except Exception as e:  # noqa: BLE001
            logger.error(f"Screen capture failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_valid(self) -> bool:
        """Whether the service was initialized successfully."""
        return self._valid

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Return screen size as (width, height)."""
        return getattr(self, "width", 0), getattr(self, "height", 0)

    def grab(self) -> Optional[np.ndarray]:
        """Capture the whole monitor as a BGR image.

        Returns:
            Reused BGR frame of shape (H, W, 3), or None on failure.
        """
        if not self._valid:
            return None

        mon = {
            "left": int(self._monitor["left"]),
            "top": int(self._monitor["top"]),
            "width": int(self._monitor["width"]),
            "height": int(self._monitor["height"]),
        }
        return self._capture_to_buffer(mon)

    def grab_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """Capture a specific region of the monitor.

        Args:
            x: Left coordinate in screen space.
            y: Top coordinate in screen space.
            width: Region width.
            height: Region height.

        Returns:
            Reused BGR frame of shape (h, w, 3), or None on failure.
        """
        if not self._valid:
            return None

        if width <= 0 or height <= 0:
            logger.warning("grab_region called with non-positive width/height")
            return None

        screen_w, screen_h = self.screen_size

        # Clamp region to screen bounds to avoid MSS errors
        left = max(0, min(x, screen_w - 1))
        top = max(0, min(y, screen_h - 1))
        right = max(left + 1, min(x + width, screen_w))
        bottom = max(top + 1, min(y + height, screen_h))

        clamped_width = right - left
        clamped_height = bottom - top

        if clamped_width <= 0 or clamped_height <= 0:
            logger.warning("grab_region: clamped region is empty, skip capture")
            return None

        mon = {
            "left": int(left),
            "top": int(top),
            "width": int(clamped_width),
            "height": int(clamped_height),
        }

        return self._capture_to_buffer(mon)

    def grab_window_by_name(self, process_name: str) -> Optional[np.ndarray]:
        """根据进程名抓取该进程主窗口图像（BGR）。"""
        if not self._valid:
            return None

        pid = find_pid_by_process_name(process_name)
        if pid is None:
            logger.error(f"grab_window_by_name: process not found: {process_name}")
            return None

        return self.grab_window_by_pid(pid)

    def grab_window_by_pid(self, pid: int) -> Optional[np.ndarray]:
        """根据进程 PID 抓取该进程主窗口图像（BGR）。"""
        if not self._valid:
            return None

        try:
            rect = self._get_window_rect_by_pid(pid)
        except Exception as e:
            logger.error(f"grab_window_by_pid: exception occurred for pid={pid}, error: {e}")
            return None

        if rect is None:
            logger.error(f"grab_window_by_pid: cannot find window for pid={pid}")
            return None

        left, top, right, bottom = rect
        width = max(1, right - left)
        height = max(1, bottom - top)

        # 直接复用你已经写好的 grab_region
        return self.grab_region(left, top, width, height)

    def _get_window_rect_by_pid(self, pid: int):
        target_hwnd = None

        def enum_handler(hwnd, _ctx):
            nonlocal target_hwnd
            if not win32gui.IsWindowVisible(hwnd):
                return
            tid, window_pid = win32process.GetWindowThreadProcessId(hwnd)
            if window_pid == pid:
                target_hwnd = hwnd

        win32gui.EnumWindows(enum_handler, None)
        if target_hwnd is None:
            return None

        left, top, right, bottom = win32gui.GetWindowRect(target_hwnd)
        return left, top, right, bottom

import win32gui
import win32process
import psutil

def find_window_rect_by_pid(pid: int):
    """根据进程 PID 找到第一个顶层窗口，并返回 (left, top, right, bottom)."""

    target_hwnd = None

    def enum_handler(hwnd, _ctx):
        nonlocal target_hwnd
        if not win32gui.IsWindowVisible(hwnd):
            return

        tid, window_pid = win32process.GetWindowThreadProcessId(hwnd)
        if window_pid == pid:
            target_hwnd = hwnd

    win32gui.EnumWindows(enum_handler, None)

    if target_hwnd is None:
        return None

    rect = win32gui.GetWindowRect(target_hwnd)  # (left, top, right, bottom)
    return rect


def find_pid_by_process_name(name: str = "WorldOfTanks.exe") -> int | None:
    # """根据进程名查 PID，大小写不敏感。"""
    name = name.lower()
    for p in psutil.process_iter(["name", "pid"]):
        try:
            if p.info["name"] and p.info["name"].lower() == name:
                return p.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def main():
    cap = CaptureService(monitor_index=1)
    if not cap.is_valid:
        logger.error("CaptureService 初始化失败")
        return

    pid = find_pid_by_process_name()  # 你从任务管理器看到的 PID
    frame = cap.grab_window_by_pid(pid)
    if frame is None:
        logger.error("抓取窗口失败，可能没找到窗口或者被保护了")
        return

    logger.info(f"frame shape: {frame.shape}")
    cv2.imshow("wot window", frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()