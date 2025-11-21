#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘控制
"""

import time
from loguru import logger

class KeyPressManager:
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
        except Exception as e:
            error_msg = f"控制服务初始化失败: {e}"
            logger.error(error_msg)
    
    def press(self, key: str) -> None:
        """按下指定按键"""
        self.keyboard_controller_.press(key)
        self.pressed_keys_.add(key)
    
    def release(self, key: str) -> None:
        """释放指定按键"""
        self.keyboard_controller_.release(key)
        self.pressed_keys_.remove(key)

    def tap(self, key: str, duration: float = 0.05) -> None:
        """点击指定按键"""
        self.keyboard_controller_.press(key)
        time.sleep(duration)
        self.keyboard_controller_.release(key)

    def press_forward(self):
        self.press('w')
    def release_forward(self):
        self.release('w')

    def press_backward(self):
        self.press('s')
    def release_backward(self):
        self.release('s')

    def press_left(self):
        self.press('a')
    def release_left(self):
        self.release('a')

    def press_right(self):
        self.press('d')
    def release_right(self):
        self.release('d')
    
    def stop_all(self) -> None:
        """停止移动（释放所有按键）"""
        try:
            for key in self.pressed_keys_:
                self.ReleaseKey(key)
        except Exception as e:
            error_msg = f"停止操作失败: {e}"
            logger.error(error_msg)

