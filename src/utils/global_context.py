#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局上下文模块

负责检测一次性的运行环境信息（屏幕分辨率、模板等级等），供其他模块复用。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from loguru import logger
import mss

DEFAULT_RESOLUTION: Tuple[int, int] = (3840, 2160)
DEFAULT_TEMPLATE_TIER = "4k"

@dataclass
class GlobalContext:
    """全局上下文，缓存启动后的临时状态。"""

    resolution: Tuple[int, int]
    template_tier: str

    def __init__(self) -> None:
        self.resolution = DEFAULT_RESOLUTION
        self.template_tier = DEFAULT_TEMPLATE_TIER
        self._detect_display_info()

    def _detect_display_info(self) -> None:
        """检测屏幕分辨率并推导模板等级。"""
        if mss is None:
            logger.warning("mss 未安装，使用默认分辨率配置")
            return

        try:
            with mss.mss() as capture:
                monitor = capture.monitors[1]  # 主显示器
                width = monitor["width"]
                height = monitor["height"]
                self.resolution = (width, height)
                self.template_tier = self._resolve_template_tier(width, height)
                logger.info(
                    f"检测到屏幕分辨率: {width}x{height}, 模板目录: {self.template_tier}"
                )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"检测屏幕分辨率失败，使用默认配置: {DEFAULT_RESOLUTION}, 原因: {exc}"
            )
            self.resolution = DEFAULT_RESOLUTION
            self.template_tier = DEFAULT_TEMPLATE_TIER

    @staticmethod
    def _resolve_template_tier(width: int, height: int) -> str:
        """根据高度判断模板目录。"""
        return "1k"
        if width == 3440 and height == 1440:
            return "2k"
        if height < 1080:
            return "1k"
        if height < 1440:
            return "2k"
        return "4k"
