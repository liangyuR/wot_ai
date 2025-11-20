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
