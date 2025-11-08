#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
瞄准辅助控制器：整合 AimAssist、鼠标控制、平滑处理
"""

from typing import Optional, Tuple
import logging

from pynput import mouse

# 统一导入机制
from wot_ai.utils.paths import setup_python_path
setup_python_path()

from wot_ai.game_modules.aim_assist.aim_assist import AimAssist

logger = logging.getLogger(__name__)


class AimController:
    """瞄准辅助控制器"""
    
    def __init__(self, aim_assist: AimAssist, smoothing_factor: float = 0.3, max_step: float = 50.0):
        """
        初始化瞄准控制器
        
        Args:
            aim_assist: AimAssist 实例
            smoothing_factor: 平滑系数（0-1）
            max_step: 最大步长（鼠标单位）
        """
        self.aim_assist_ = aim_assist
        self.smoothing_factor_ = smoothing_factor
        self.max_step_ = max_step
        self.mouse_controller_ = mouse.Controller()
        
        # 平滑状态
        self.current_smoothed_x_ = 0.0
        self.current_smoothed_y_ = 0.0
        
        logger.info(f"瞄准控制器初始化: 平滑系数={smoothing_factor}, 最大步长={max_step}")
    
    def ProcessTarget(self, target_x: int, target_y: int, 
                     current_crosshair_x: Optional[int] = None,
                     current_crosshair_y: Optional[int] = None) -> Tuple[float, float]:
        """
        处理目标，计算并执行鼠标移动
        
        Args:
            target_x: 目标X坐标
            target_y: 目标Y坐标
            current_crosshair_x: 当前准星X坐标（如果None则使用屏幕中心）
            current_crosshair_y: 当前准星Y坐标（如果None则使用屏幕中心）
        
        Returns:
            (实际移动的dx, dy)
        """
        # 计算鼠标移动量
        mouse_x, mouse_y = self.aim_assist_.TargetToMouseMovement(
            target_x, target_y,
            current_crosshair_x=current_crosshair_x,
            current_crosshair_y=current_crosshair_y
        )
        
        # 平滑处理
        dx, dy = self.aim_assist_.SmoothMovement(
            target_mouse_x=mouse_x,
            target_mouse_y=mouse_y,
            current_mouse_x=self.current_smoothed_x_,
            current_mouse_y=self.current_smoothed_y_,
            smoothing_factor=self.smoothing_factor_,
            max_step=self.max_step_
        )
        
        # 更新累计移动状态
        self.current_smoothed_x_ += dx
        self.current_smoothed_y_ += dy
        
        # 执行鼠标移动
        self.mouse_controller_.move_rel(int(dx), int(dy))
        
        return dx, dy
    
    def ResetSmoothing(self):
        """重置平滑状态"""
        self.current_smoothed_x_ = 0.0
        self.current_smoothed_y_ = 0.0
    
    def UpdateSmoothingConfig(self, smoothing_factor: float, max_step: float):
        """
        更新平滑配置
        
        Args:
            smoothing_factor: 新的平滑系数
            max_step: 新的最大步长
        """
        self.smoothing_factor_ = smoothing_factor
        self.max_step_ = max_step
    
    def UpdateAimAssist(self, aim_assist: AimAssist):
        """
        更新 AimAssist 实例（用于热更新配置）
        
        Args:
            aim_assist: 新的 AimAssist 实例
        """
        self.aim_assist_ = aim_assist
        self.ResetSmoothing()

