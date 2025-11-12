#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay配置：定义透明浮窗的配置参数
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OverlayConfig:
    """Overlay 配置"""
    width: int = 1920
    height: int = 1080
    pos_x: int = 0                 # 窗口X坐标
    pos_y: int = 0                 # 窗口Y坐标
    fps: int = 30                  # 帧率：用来做高/中/低性能
    alpha: int = 180               # 0-255 全局透明度（180 差不多够用）
    title: str = "WOT AI Overlay"  # Overlay 窗口标题
    target_window_title: Optional[str] = None  # 绑定的游戏窗口标题，如 "World of Tanks"
    click_through: bool = True     # 鼠标穿透

