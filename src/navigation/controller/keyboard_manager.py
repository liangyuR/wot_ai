#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Set

from loguru import logger

class KeyPressManager:
    """键盘控制服务（使用 WASD 控制移动和转向）"""

    def __init__(self) -> None:
        """初始化控制服务"""
        try:
            from pynput import keyboard

            self.keyboard_controller_ = keyboard.Controller()
            # 按键状态跟踪
            self.pressed_keys_: Set[str] = set()
            logger.info("KeyPressManager 初始化完成（仅键盘控制）")
        except Exception as e:  # pragma: no cover - 极端环境保护
            self.keyboard_controller_ = None
            self.pressed_keys_ = set()
            logger.error(f"KeyPressManager 初始化失败: {e}")

    # ---- primitive operations ----

    def press(self, key: str) -> None:
        """按下指定按键（带状态去重）"""
        if self.keyboard_controller_ is None:
            return
        if key in self.pressed_keys_:
            return
        self.keyboard_controller_.press(key)
        self.pressed_keys_.add(key)

    def release(self, key: str) -> None:
        """释放指定按键（安全删除）"""
        if self.keyboard_controller_ is None:
            return
        if key not in self.pressed_keys_:
            return
        self.keyboard_controller_.release(key)
        self.pressed_keys_.discard(key)

    def tap(self, key: str, duration: float = 0.05) -> None:
        """点击指定按键（用于开火、切换等瞬时操作）"""
        if self.keyboard_controller_ is None:
            return
        self.keyboard_controller_.press(key)
        time.sleep(duration)
        self.keyboard_controller_.release(key)

    # ---- semantic helpers for movement ----

    def press_forward(self) -> None:
        self.press("w")

    def release_forward(self) -> None:
        self.release("w")

    def press_backward(self) -> None:
        self.press("s")

    def release_backward(self) -> None:
        self.release("s")

    def press_turn_left(self) -> None:
        self.press("a")

    def release_turn_left(self) -> None:
        self.release("a")

    def press_turn_right(self) -> None:
        self.press("d")

    def release_turn_right(self) -> None:
        self.release("d")

    def stop_all(self) -> None:
        """停止移动（释放所有已按下按键）"""
        if self.keyboard_controller_ is None:
            return
        try:
            # 拷贝一份，避免迭代时修改集合
            for key in list(self.pressed_keys_):
                self.keyboard_controller_.release(key)
            self.pressed_keys_.clear()
        except Exception as e:  # pragma: no cover - 防御性
            logger.error(f"KeyPressManager.stop_all 失败: {e}")
