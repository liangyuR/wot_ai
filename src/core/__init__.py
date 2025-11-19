#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心控制逻辑模块

包含：
- 状态机（StateMachine）
- 动作层（Actions）
- 地图名称识别（MapNameDetector）
- 坦克选择（TankSelector）
- 战斗任务（BattleTask）
- 任务管理器（TaskManager）
- AI控制器（AIController）
"""

# 核心模块导出
from .state_machine import StateMachine, GameState
from .actions import screenshot
from .tank_selector import TankSelector
from .battle_task import BattleTask
from .task_manager import TaskManager
from .ai_controller import AIController

__all__ = [
    'StateMachine',
    'GameState',
    'screenshot',
    'TankSelector',
    'BattleTask',
    'TaskManager',
    'AIController',
]

