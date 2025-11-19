#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作层模块（简化版）

直接使用 pyautogui API，只提供简单的等待和状态检查功能。
"""

import time
from typing import Optional, Tuple
from loguru import logger
import pyautogui
import mss
import numpy as np
import cv2
from pynput import keyboard



def wait(seconds: float) -> None:
    """
    等待指定时间
    
    Args:
        seconds: 等待时间（秒）
    """
    time.sleep(seconds)


def wait_state(
    state_machine,
    target_state,
    timeout: float = 10.0,
    check_interval: float = 0.5
) -> bool:
    """
    等待状态机切换到目标状态
    
    Args:
        state_machine: StateMachine 实例
        target_state: 目标状态（GameState枚举）
        timeout: 超时时间（秒）
        check_interval: 检查间隔（秒）
    
    Returns:
        是否在超时前达到目标状态
    """
    return state_machine.wait_state(target_state, timeout, check_interval)


def screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """
    截取屏幕截图
    
    Args:
        region: 可选区域 (left, top, width, height)，None表示全屏
    
    Returns:
        BGR格式的numpy数组，如果失败则返回None
    """
    if mss is None or np is None or cv2 is None:
        logger.error("mss/numpy/opencv 未安装，无法截图")
        return None
    
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


def screenshot_with_key_hold(
    key: keyboard.Key,
    hold_duration: float = 2.0,
    warmup: float = 0.0
) -> Optional[np.ndarray]:
    """
    按住指定按键后截图

    Args:
        key: 按键字符（如 'b'）
        hold_duration: 按键保持时间（秒）
        warmup: 按键后等待界面稳定的时间（秒）

    Returns:
        BGR格式的numpy数组截图，如果失败则返回None
    """
    kb = keyboard.Controller()
    
    def _key_down():
        try:
            kb.press(key)
            return True
        except Exception as exc:
            logger.error(f"pynput 按键失败 {key}: {exc}")
            return False
    
    def _key_up():
        try:
            kb.release(key)
        except Exception as exc:
            logger.error(f"释放按键 {key} 失败: {exc}")
    
    if not _key_down():
        return None
    
    try:
        logger.info(f"按下按键 {key} 并保持 {hold_duration} 秒后截图")
        if warmup > 0:
            wait(warmup)
        remain = max(0.0, hold_duration - warmup)
        if remain > 0:
            wait(remain)
        frame = screenshot()
        return frame
    except Exception as exc:
        logger.error(f"按住 {key} 截图失败: {exc}")
        return None
    finally:
        _key_up()
