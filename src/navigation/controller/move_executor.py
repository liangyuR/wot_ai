# navigation_executor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from loguru import logger
from typing import Optional

from src.navigation.controller.keyboard_manager import KeyPressManager  # 你自己的路径

class MoveExecutor:
    """
    移动执行器
    """
    def __init__(self) -> None:
        self.key_press_manager = KeyPressManager()

    def stop_all(self) -> None:
        """释放所有移动相关按键"""
        self.key_press_manager.stop_all()

    def start_forward(self) -> None:
        """开始持续前进：按下 W，确保 S 松开"""
        self.key_press_manager.release_backward()
        self.key_press_manager.press_forward()

    def stop_forward(self) -> None:
        """停止前进：松开 W"""
        self._release("w")

    def start_backward(self) -> None:
        """开始持续后退：按下 S，确保 W 松开"""
        self._release("w")
        self._press("s")

    def stop_backward(self) -> None:
        """停止后退：松开 S"""
        self._release("s")

    # 左右转
    def start_turn_left(self) -> None:
        """开始持续左转：按下 A，松开 D"""
        self._release("d")
        self._press("a")

    def start_turn_right(self) -> None:
        """开始持续右转：按下 D，松开 A"""
        self._release("a")
        self._press("d")

    def stop_turn(self) -> None:
        """停止转向：松开 A/D"""
        self._release("a")
        self._release("d")

    # 组合动作
    def ensure_moving_forward(self) -> None:
        """
        确保“正在向前移动”：
        - 如果 W 未按下，则按下
        - 如果 S 正在按下，先松开 S
        """
        self.start_forward()

    def stop_movement(self) -> None:
        """
        停止一切移动（前进/后退/转向）
        """
        self.stop_all()
