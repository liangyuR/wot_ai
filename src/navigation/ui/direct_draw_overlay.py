#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectDrawOverlay 已退役：保留空实现以兼容既有代码路径。
"""

from typing import List, Optional, Tuple, Any

from loguru import logger


class DirectDrawOverlay:
    """空实现，避免创建任何窗口或 GDI 资源。"""

    def __init__(self, width: int, height: int,
                 pos_x: int = 0, pos_y: int = 0,
                 fps: int = 30) -> None:
        self.width_ = width
        self.height_ = height
        self.pos_x_ = pos_x
        self.pos_y_ = pos_y
        self.fps_ = fps
        self.visible_ = False
        logger.debug(
            "DirectDrawOverlay 已禁用（%dx%d @ (%d,%d))",
            width, height, pos_x, pos_y
        )

    def Start_(self) -> None:
        self.visible_ = True

    def DrawPath(
        self,
        minimap: Optional[Any],
        path: List[Tuple[int, int]],
        detections: dict,
        minimap_size: Tuple[int, int] = None,
        grid_size: Tuple[int, int] = None,
        mask: Any = None,
    ) -> None:
        return

    def SetPosition(self, x: int, y: int) -> None:
        self.pos_x_ = x
        self.pos_y_ = y

    def SetFps(self, fps: int) -> None:
        self.fps_ = fps

    def Hide(self) -> None:
        self.visible_ = False

    def Show(self) -> None:
        self.visible_ = True

    def IsVisible(self) -> bool:
        return self.visible_

    def Close(self) -> None:
        self.visible_ = False

