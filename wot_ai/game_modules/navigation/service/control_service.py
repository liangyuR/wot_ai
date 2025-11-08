#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鼠标键盘控制服务：封装游戏操作逻辑
"""

import time
import math
from typing import Optional, Tuple
# 统一导入机制
from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple
setup_python_path()

SetupLogger = None
logger_module, _ = try_import_multiple([
    'wot_ai.game_modules.common.utils.logger',
    'game_modules.common.utils.logger',
    'common.utils.logger',
    'yolo.utils.logger'
])
if logger_module is not None:
    SetupLogger = getattr(logger_module, 'SetupLogger', None)

if SetupLogger is None:
    from ..common.utils.logger import SetupLogger

logger = SetupLogger(__name__)


class ControlService:
    """鼠标键盘控制服务"""
    
    def __init__(self, mouse_sensitivity: float = 1.0, calibration_factor: float = 150.0):
        """
        初始化控制服务
        
        Args:
            mouse_sensitivity: 鼠标灵敏度
            calibration_factor: 标定系数
        """
        try:
            from pynput import mouse, keyboard
            self.mouse_controller_ = mouse.Controller()
            self.keyboard_controller_ = keyboard.Controller()
            self.mouse_sensitivity_ = mouse_sensitivity
            self.calibration_factor_ = calibration_factor
            logger.info(f"控制服务初始化: 灵敏度={mouse_sensitivity}, 标定系数={calibration_factor}")
        except ImportError:
            logger.error("pynput 未安装，请运行: pip install pynput")
            raise
        except Exception as e:
            logger.error(f"控制服务初始化失败: {e}")
            raise
    
    def RotateToward(self, target_pos: Tuple[float, float], current_pos: Tuple[float, float]) -> None:
        """
        转向目标位置
        
        Args:
            target_pos: 目标位置 (x, y)（小地图坐标）
            current_pos: 当前位置 (x, y)（小地图坐标）
        """
        # 计算方向向量
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if dx == 0 and dy == 0:
            return
        
        # 计算角度（弧度）
        angle = math.atan2(dy, dx)
        
        # 转换为鼠标移动（需要根据实际游戏调整）
        # 这里使用简化的转换：角度直接转换为鼠标移动
        mouse_dx = math.cos(angle) * self.calibration_factor_ * self.mouse_sensitivity_
        mouse_dy = math.sin(angle) * self.calibration_factor_ * self.mouse_sensitivity_
        
        # 移动鼠标
        self.mouse_controller_.move(int(mouse_dx), int(mouse_dy))
    
    def MoveForward(self, duration: float = 1.0) -> None:
        """
        前进
        
        Args:
            duration: 持续时间（秒）
        """
        try:
            from pynput.keyboard import Key
            # 按下W键
            self.keyboard_controller_.press('w')
            time.sleep(duration)
            # 释放W键
            self.keyboard_controller_.release('w')
        except Exception as e:
            logger.error(f"前进操作失败: {e}")
    
    def MoveBackward(self, duration: float = 1.0) -> None:
        """
        后退
        
        Args:
            duration: 持续时间（秒）
        """
        try:
            # 按下S键
            self.keyboard_controller_.press('s')
            time.sleep(duration)
            # 释放S键
            self.keyboard_controller_.release('s')
        except Exception as e:
            logger.error(f"后退操作失败: {e}")
    
    def Stop(self) -> None:
        """停止移动（释放所有按键）"""
        try:
            from pynput.keyboard import Key
            # 释放常用移动键
            for key in ['w', 's', 'a', 'd']:
                try:
                    self.keyboard_controller_.release(key)
                except:
                    pass
        except Exception as e:
            logger.error(f"停止操作失败: {e}")

