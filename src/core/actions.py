#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作层模块（简化版）

直接使用 pyautogui API，只提供简单的等待和状态检查功能。
"""

import time
from typing import Optional, Tuple
from loguru import logger
import mss
import numpy as np
import cv2
from pynput import keyboard

class ScreenAction:
    """屏幕操作类：封装截图和按键操作"""
    
    def __init__(self):
        """初始化屏幕操作类"""
        self.keyboard_controller_ = keyboard.Controller()
    
    def Screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        截取屏幕截图
        
        Args:
            region: 可选区域 (left, top, width, height)，None表示全屏
        
        Returns:
            BGR格式的numpy数组，如果失败则返回None
        """
        try:
            with mss.mss() as sct:
                if region is None:
                    # 全屏截图
                    monitor = sct.monitors[1]  # 主显示器
                    screenshot = sct.grab(monitor)
                else:
                    # 区域截图
                    left, top, width, height = region
                    monitor = {
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height
                    }
                    screenshot = sct.grab(monitor)
                
                # 转换为numpy数组（BGR格式）
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None
    
    def ScreenshotWithKeyHold(
        self,
        key: keyboard.Key,
        hold_duration: float = 2.0,
        warmup: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        按住指定按键后截图

        Args:
            key: 按键对象
            hold_duration: 按键保持时间（秒）
            warmup: 按键后等待界面稳定的时间（秒）

        Returns:
            BGR格式的numpy数组截图，如果失败则返回None
        """
        if not self._KeyDown_(key):
            return None
        
        try:
            logger.info(f"按下按键 {key} 并保持 {hold_duration} 秒后截图")
            if warmup > 0:
                time.sleep(warmup)
            remain = max(0.0, hold_duration - warmup)
            if remain > 0:
                time.sleep(remain)
            frame = self.Screenshot()
            return frame
        except Exception as exc:
            logger.error(f"按住 {key} 截图失败: {exc}")
            return None
        finally:
            self._KeyUp_(key)
    
    def _KeyDown_(self, key: keyboard.Key) -> bool:
        """
        按下按键
        
        Args:
            key: 按键对象
        
        Returns:
            是否成功按下
        """
        try:
            self.keyboard_controller_.press(key)
            return True
        except Exception as exc:
            logger.error(f"pynput 按键失败 {key}: {exc}")
            return False
    
    def _KeyUp_(self, key: keyboard.Key) -> None:
        """
        释放按键
        
        Args:
            key: 按键对象
        """
        try:
            self.keyboard_controller_.release(key)
        except Exception as exc:
            logger.error(f"释放按键 {key} 失败: {exc}")


# 保持向后兼容的函数接口
_screen_action_instance_ = ScreenAction()


def screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """
    截取屏幕截图（向后兼容接口）
    
    Args:
        region: 可选区域 (left, top, width, height)，None表示全屏
    
    Returns:
        BGR格式的numpy数组，如果失败则返回None
    """
    return _screen_action_instance_.Screenshot(region)


def screenshot_with_key_hold(
    key: keyboard.Key,
    hold_duration: float = 2.0,
    warmup: float = 0.0
) -> Optional[np.ndarray]:
    """
    按住指定按键后截图（向后兼容接口）

    Args:
        key: 按键对象
        hold_duration: 按键保持时间（秒）
        warmup: 按键后等待界面稳定的时间（秒）

    Returns:
        BGR格式的numpy数组截图，如果失败则返回None
    """
    return _screen_action_instance_.ScreenshotWithKeyHold(key, hold_duration, warmup)
