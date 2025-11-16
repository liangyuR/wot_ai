#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI 自动化控制模块

功能：
- 通过模板匹配和鼠标控制自动执行游戏 UI 操作
- 实现从车库到战斗的完整流程自动化
"""

from .matcher_pyautogui import match_template
from .actions import UIActions
from .ui_flow import UIFlow

__all__ = [
    'match_template',
    'UIActions',
    'UIFlow',
]

