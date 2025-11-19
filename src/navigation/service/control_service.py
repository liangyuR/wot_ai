#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鼠标键盘控制服务：封装游戏操作逻辑
"""

# 标准库导入
from typing import Optional, Tuple
import math
import time

# 本地模块导入
from loguru import logger


class ControlService:
    """键盘控制服务（使用WASD控制移动和转向）"""
    
    def __init__(self):
        """
        初始化控制服务
        """
        try:
            from pynput import keyboard
            self.keyboard_controller_ = keyboard.Controller()
            # 按键状态跟踪
            self.pressed_keys_ = set()
            logger.info("控制服务初始化完成（仅键盘控制）")
        except ImportError as e:
            error_msg = "pynput 未安装，请运行: pip install pynput"
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"控制服务初始化失败: {e}"
            logger.error(error_msg)
    
    def CalculateAngleDifference(self, current_heading: float, target_heading: float) -> Tuple[float, int]:
        """
        计算两个角度之间的最小差值（考虑360度循环）
        
        Args:
            current_heading: 当前朝向角度（弧度）
            target_heading: 目标朝向角度（弧度）
        
        Returns:
            (角度差（弧度）, 转向方向：-1左转，1右转，0不需要转向)
        """
        # 归一化角度到 [0, 2π)
        current_heading = current_heading % (2 * math.pi)
        target_heading = target_heading % (2 * math.pi)
        
        # 计算角度差
        diff = target_heading - current_heading
        
        # 处理跨越0度的情况
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
        
        # 确定转向方向
        if abs(diff) < 0.001:  # 几乎不需要转向
            return (0.0, 0)
        elif diff > 0:
            return (diff, 1)  # 右转
        else:
            return (abs(diff), -1)  # 左转
    
    def AdjustDirectionWithAD(self, current_heading: float, target_heading: float, angle_threshold: float = math.radians(5.0)) -> None:
        """
        根据角度差使用A/D键调整方向
        
        Args:
            current_heading: 当前朝向角度（弧度）
            target_heading: 目标朝向角度（弧度）
            angle_threshold: 角度阈值（弧度），小于此值时不转向
        """
        angle_diff, turn_direction = self.CalculateAngleDifference(current_heading, target_heading)
        
        # 先释放所有转向键
        self.StopLeft()
        self.StopRight()
        
        # 如果角度差大于阈值，按下相应的转向键
        if angle_diff > angle_threshold:
            if turn_direction == -1:  # 左转
                self.StartLeft()
            elif turn_direction == 1:  # 右转
                self.StartRight()
    
    def RotateToward(
        self,
        target_pos: Tuple[float, float],
        current_pos: Tuple[float, float],
        current_heading: Optional[float] = None,
    ) -> None:
        """
        转向目标位置（使用A/D键）

        Args:
            target_pos: 目标位置 (x, y)（小地图坐标）
            current_pos: 当前位置 (x, y)（小地图坐标）
            current_heading: 当前朝向角度（弧度），未知时自动估计
        """
        # 计算方向向量
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        if dx == 0 and dy == 0:
            return

        # 计算目标方向角度（弧度）
        target_heading = math.atan2(dy, dx)

        if current_heading is None:
            current_heading = target_heading

        # 使用A/D键调整方向
        self.AdjustDirectionWithAD(current_heading, target_heading)

    def TapKey(self, key: str, duration: float = 0.05) -> None:
        """短按按键（用于 PWM 控制）"""

        if duration <= 0:
            return
        self.PressKey(key)
        time.sleep(duration)
        self.ReleaseKey(key)
    
    def MoveForward(self, duration: float = 1.0) -> None:
        """
        前进
        
        Args:
            duration: 持续时间（秒）
        """
        try:
            # 按下W键
            self.keyboard_controller_.press('w')
            time.sleep(duration)
            # 释放W键
            self.keyboard_controller_.release('w')
        except Exception as e:
            error_msg = f"前进操作失败: {e}"
            logger.error(error_msg)
    
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
            error_msg = f"后退操作失败: {e}"
            logger.error(error_msg)
    
    def PressKey(self, key: str) -> None:
        """
        按下按键（如果未按下）
        
        Args:
            key: 按键字符（如 'w', 's', 'a', 'd'）
        """
        if key not in self.pressed_keys_:
            try:
                # logger.debug(f"准备按下按键: {key}")
                self.keyboard_controller_.press(key)
                self.pressed_keys_.add(key)
                # logger.debug(f"成功按下按键: {key}")
            except Exception as e:
                # logger.error(f"按下按键失败 {key}: {e}")
                import traceback
                traceback.print_exc()
    
    def ReleaseKey(self, key: str) -> None:
        """
        释放按键（如果已按下）
        
        Args:
            key: 按键字符（如 'w', 's', 'a', 'd'）
        """
        if key in self.pressed_keys_:
            try:
                self.keyboard_controller_.release(key)
                self.pressed_keys_.discard(key)
                # logger.debug(f"释放按键: {key}")
            except Exception as e:
                logger.warning(f"释放按键失败 {key}: {e}")
    
    def IsKeyPressed(self, key: str) -> bool:
        """
        检查按键是否已按下
        
        Args:
            key: 按键字符
        
        Returns:
            是否已按下
        """
        return key in self.pressed_keys_
    
    def StartForward(self) -> None:
        """开始持续前进（按下W键）"""
        self.PressKey('w')
    
    def StopForward(self) -> None:
        """停止前进（释放W键）"""
        self.ReleaseKey('w')
    
    def StartLeft(self) -> None:
        """开始持续左转（按下A键）"""
        self.PressKey('a')
    
    def StopLeft(self) -> None:
        """停止左转（释放A键）"""
        self.ReleaseKey('a')
    
    def StartRight(self) -> None:
        """开始持续右转（按下D键）"""
        self.PressKey('d')
    
    def StopRight(self) -> None:
        """停止右转（释放D键）"""
        self.ReleaseKey('d')
    
    def StartBackward(self) -> None:
        """开始持续后退（按下S键）"""
        self.PressKey('s')
    
    def StopBackward(self) -> None:
        """停止后退（释放S键）"""
        self.ReleaseKey('s')
    
    def Stop(self) -> None:
        """停止移动（释放所有按键）"""
        try:
            # 释放所有已按下的移动键
            keys_to_release = list(self.pressed_keys_)
            for key in keys_to_release:
                self.ReleaseKey(key)
        except Exception as e:
            error_msg = f"停止操作失败: {e}"
            logger.error(error_msg)
            # 停止操作失败不抛出异常，避免影响清理流程

