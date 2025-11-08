#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热键管理器：管理键盘热键监听和回调
"""

from typing import Callable, Optional, Dict
import logging
from threading import Lock

from pynput import keyboard
from pynput.keyboard import Key, Listener

logger = logging.getLogger(__name__)


class HotkeyManager:
    """热键管理器"""
    
    def __init__(self):
        """初始化热键管理器"""
        self.listener_ = None
        self.hotkey_callbacks_: Dict[str, Callable] = {}
        self.key_map_: Dict[str, Key] = {}
        self.lock_ = Lock()
        self.is_running_ = False
    
    def RegisterHotkey(self, hotkey_name: str, key_name: str, callback: Callable):
        """
        注册热键
        
        Args:
            hotkey_name: 热键名称（如 'toggle', 'exit'）
            key_name: 按键名称（如 'f8', 'esc'）
            callback: 回调函数
        """
        with self.lock_:
            self.hotkey_callbacks_[hotkey_name] = callback
            self.key_map_[hotkey_name] = self._ParseKey(key_name)
            logger.info(f"注册热键: {hotkey_name} = {key_name}")
    
    def _ParseKey(self, key_name: str) -> Key:
        """
        解析按键名称
        
        Args:
            key_name: 按键名称（如 'f8', 'esc', 'ctrl'）
        
        Returns:
            Key 对象
        """
        key_name_lower = key_name.lower()
        
        # 功能键
        if key_name_lower.startswith('f') and key_name_lower[1:].isdigit():
            fn_num = int(key_name_lower[1:])
            fn_keys = {
                1: Key.f1, 2: Key.f2, 3: Key.f3, 4: Key.f4,
                5: Key.f5, 6: Key.f6, 7: Key.f7, 8: Key.f8,
                9: Key.f9, 10: Key.f10, 11: Key.f11, 12: Key.f12
            }
            if fn_num in fn_keys:
                return fn_keys[fn_num]
        
        # 特殊键
        special_keys = {
            'esc': Key.esc,
            'escape': Key.esc,
            'space': Key.space,
            'enter': Key.enter,
            'tab': Key.tab,
            'backspace': Key.backspace,
            'delete': Key.delete,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'home': Key.home,
            'end': Key.end,
            'page_up': Key.page_up,
            'page_down': Key.page_down,
        }
        
        if key_name_lower in special_keys:
            return special_keys[key_name_lower]
        
        # 默认返回字符键
        try:
            return Key[key_name_lower]
        except KeyError:
            logger.warning(f"无法解析按键: {key_name}，使用字符键")
            return Key.from_char(key_name_lower)
    
    def _OnKeyPress(self, key):
        """按键按下回调"""
        try:
            with self.lock_:
                # 检查是否匹配任何注册的热键
                for hotkey_name, registered_key in self.key_map_.items():
                    if key == registered_key:
                        callback = self.hotkey_callbacks_.get(hotkey_name)
                        if callback:
                            try:
                                callback()
                            except Exception as e:
                                logger.error(f"热键回调执行失败 ({hotkey_name}): {e}")
        except Exception as e:
            logger.debug(f"处理按键事件失败: {e}")
    
    def Start(self):
        """启动热键监听"""
        if self.is_running_:
            logger.warning("热键监听已在运行")
            return
        
        self.listener_ = Listener(on_press=self._OnKeyPress)
        self.listener_.start()
        self.is_running_ = True
        logger.info("热键监听已启动")
    
    def Stop(self):
        """停止热键监听"""
        if not self.is_running_:
            return
        
        if self.listener_:
            self.listener_.stop()
            self.listener_ = None
        
        self.is_running_ = False
        logger.info("热键监听已停止")
    
    def UpdateHotkey(self, hotkey_name: str, key_name: str):
        """
        更新热键（运行时修改）
        
        Args:
            hotkey_name: 热键名称
            key_name: 新的按键名称
        """
        with self.lock_:
            if hotkey_name in self.hotkey_callbacks_:
                self.key_map_[hotkey_name] = self._ParseKey(key_name)
                logger.info(f"更新热键: {hotkey_name} = {key_name}")
            else:
                logger.warning(f"热键不存在，无法更新: {hotkey_name}")

