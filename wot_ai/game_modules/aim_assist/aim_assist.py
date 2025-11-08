#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
瞄准辅助模块
将YOLO检测到的目标弱点坐标转换为鼠标移动指令
"""

import math
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AimAssist:
    """瞄准辅助系统：将目标像素坐标转换为鼠标移动"""
    
    def __init__(self, 
                 screen_width: int,
                 screen_height: int,
                 horizontal_fov: float = 90.0,  # 水平视场角（度）
                 vertical_fov: Optional[float] = None,  # 垂直视场角，如果None则自动计算
                 mouse_sensitivity: float = 1.0,  # 鼠标灵敏度（需要标定）
                 calibration_factor: Optional[float] = None):  # 标定系数（角度/鼠标单位）
        """
        初始化瞄准辅助
        
        Args:
            screen_width: 屏幕宽度（像素）
            screen_height: 屏幕高度（像素）
            horizontal_fov: 水平视场角（度），通常在游戏设置中可查
            vertical_fov: 垂直视场角（度），如果None则根据宽高比计算
            mouse_sensitivity: 鼠标灵敏度（游戏内设置）
            calibration_factor: 标定系数（度/鼠标单位），需要实际测量
        """
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.h_fov_deg = horizontal_fov
        self.h_fov_rad = math.radians(horizontal_fov)
        
        # 计算垂直FOV（基于宽高比）
        if vertical_fov is None:
            aspect_ratio = screen_width / screen_height
            # v_fov = 2 * arctan(tan(h_fov/2) / aspect_ratio)
            self.v_fov_rad = 2 * math.atan(math.tan(self.h_fov_rad / 2) / aspect_ratio)
            self.v_fov_deg = math.degrees(self.v_fov_rad)
        else:
            self.v_fov_deg = vertical_fov
            self.v_fov_rad = math.radians(vertical_fov)
        
        self.mouse_sensitivity = mouse_sensitivity
        
        # 标定系数（需要实际测量）
        # 含义：每1度视角需要多少鼠标单位（counts/degree）
        # 如果None，使用默认估算
        if calibration_factor is None:
            # 默认估算：假设中等灵敏度下，约需要100-200鼠标单位/度
            # 这个值需要实际标定
            self.calibration_factor = 150.0 / mouse_sensitivity
            logger.warning(f"使用默认标定系数 {self.calibration_factor:.2f}，建议进行实际标定")
        else:
            self.calibration_factor = calibration_factor
        
        # 屏幕中心
        self.center_x = screen_width / 2.0
        self.center_y = screen_height / 2.0
        
        logger.info(f"瞄准系统初始化: 屏幕={screen_width}x{screen_height}, "
                   f"FOV={horizontal_fov:.1f}°x{self.v_fov_deg:.1f}°, "
                   f"灵敏度={mouse_sensitivity}, 标定系数={self.calibration_factor:.2f}")
    
    def PixelToAngle(self, dx_px: float, dy_px: float) -> Tuple[float, float]:
        """
        将像素偏移转换为角度偏移
        
        Args:
            dx_px: 水平像素偏移（像素）
            dy_px: 垂直像素偏移（像素）
            
        Returns:
            (水平角度偏移, 垂直角度偏移) 单位：度
        """
        # 方法1：精确公式（适用于任意角度）
        # 水平角度
        half_width = self.screen_w / 2.0
        if abs(dx_px) > 0:
            # 计算目标点在屏幕上的归一化位置（相对于中心）
            x_norm = dx_px / half_width
            # 使用精确公式
            angle_x_rad = 2 * math.atan(x_norm * math.tan(self.h_fov_rad / 2))
        else:
            angle_x_rad = 0.0
        
        # 垂直角度
        half_height = self.screen_h / 2.0
        if abs(dy_px) > 0:
            y_norm = dy_px / half_height
            angle_y_rad = 2 * math.atan(y_norm * math.tan(self.v_fov_rad / 2))
        else:
            angle_y_rad = 0.0
        
        angle_x_deg = math.degrees(angle_x_rad)
        angle_y_deg = math.degrees(angle_y_rad)
        
        return angle_x_deg, angle_y_deg
    
    def AngleToMouseUnits(self, angle_x_deg: float, angle_y_deg: float) -> Tuple[float, float]:
        """
        将角度偏移转换为鼠标移动单位
        
        Args:
            angle_x_deg: 水平角度偏移（度）
            angle_y_deg: 垂直角度偏移（度）
            
        Returns:
            (鼠标水平移动单位, 鼠标垂直移动单位)
        """
        # 考虑灵敏度
        mouse_x = angle_x_deg * self.calibration_factor * self.mouse_sensitivity
        mouse_y = angle_y_deg * self.calibration_factor * self.mouse_sensitivity
        
        return mouse_x, mouse_y
    
    def TargetToMouseMovement(self, 
                             target_x: int, 
                             target_y: int,
                             current_crosshair_x: Optional[int] = None,
                             current_crosshair_y: Optional[int] = None) -> Tuple[float, float]:
        """
        完整的转换流程：目标像素坐标 -> 鼠标移动单位
        
        Args:
            target_x: 目标像素X坐标
            target_y: 目标像素Y坐标
            current_crosshair_x: 当前准星X坐标（如果None则使用屏幕中心）
            current_crosshair_y: 当前准星Y坐标（如果None则使用屏幕中心）
            
        Returns:
            (鼠标水平移动单位, 鼠标垂直移动单位)
        """
        # 1. 计算像素偏移
        crosshair_x = current_crosshair_x if current_crosshair_x is not None else self.center_x
        crosshair_y = current_crosshair_y if current_crosshair_y is not None else self.center_y
        
        dx_px = target_x - crosshair_x
        dy_px = target_y - crosshair_y
        
        # 2. 转换为角度偏移
        angle_x, angle_y = self.PixelToAngle(dx_px, dy_px)
        
        # 3. 转换为鼠标移动单位
        mouse_x, mouse_y = self.AngleToMouseUnits(angle_x, angle_y)
        
        return mouse_x, mouse_y
    
    def SmoothMovement(self, 
                     target_mouse_x: float,
                     target_mouse_y: float,
                     current_mouse_x: float = 0.0,
                     current_mouse_y: float = 0.0,
                     smoothing_factor: float = 0.3,
                     max_step: float = 50.0) -> Tuple[float, float]:
        """
        平滑鼠标移动（EMA平滑 + 最大步长限制）
        
        Args:
            target_mouse_x: 目标鼠标移动X
            target_mouse_y: 目标鼠标移动Y
            current_mouse_x: 当前累计移动X（用于平滑）
            current_mouse_y: 当前累计移动Y（用于平滑）
            smoothing_factor: 平滑系数（0-1，越小越平滑）
            max_step: 单次最大移动步长
            
        Returns:
            (平滑后的鼠标移动X, 平滑后的鼠标移动Y)
        """
        # EMA平滑
        smoothed_x = smoothing_factor * target_mouse_x + (1 - smoothing_factor) * current_mouse_x
        smoothed_y = smoothing_factor * target_mouse_y + (1 - smoothing_factor) * current_mouse_y
        
        # 计算本次实际移动量
        dx = smoothed_x - current_mouse_x
        dy = smoothed_y - current_mouse_y
        
        # 限制最大步长
        distance = math.hypot(dx, dy)
        if distance > max_step:
            scale = max_step / distance
            dx *= scale
            dy *= scale
        
        return dx, dy


class CalibrationTool:
    """标定工具：用于测量实际的"角度/鼠标单位"比例"""
    
    @staticmethod
    def MeasureCalibrationFactor(aim_assist: AimAssist,
                                 test_distance_px: float = 100.0,
                                 test_mouse_units: Optional[float] = None) -> Optional[float]:
        """
        测量标定系数
        
        方法：在屏幕上标记一个已知距离的点，移动鼠标直到准星对准该点，
        记录鼠标移动单位，计算比例。
        
        Args:
            aim_assist: 瞄准辅助实例
            test_distance_px: 测试距离（像素）
            test_mouse_units: 实际测量的鼠标移动单位（如果None则需要手动输入）
            
        Returns:
            标定系数（度/鼠标单位），如果无法测量则返回None
        """
        # 计算对应角度
        angle_x, _ = aim_assist.PixelToAngle(test_distance_px, 0.0)
        
        if test_mouse_units is None:
            logger.warning("需要手动测量鼠标移动单位。"
                         f"在屏幕上标记一个距离中心{test_distance_px}像素的点，"
                         "移动鼠标直到准星对准，记录鼠标移动单位。")
            return None
        
        # 计算比例
        calibration_factor = angle_x / test_mouse_units
        logger.info(f"标定完成: {test_distance_px}px = {angle_x:.2f}° = {test_mouse_units}鼠标单位")
        logger.info(f"标定系数: {calibration_factor:.2f} 度/鼠标单位")
        
        return calibration_factor

