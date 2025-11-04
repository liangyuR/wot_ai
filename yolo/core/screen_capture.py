#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
屏幕捕获抽象类
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ScreenCapture(ABC):
    """屏幕捕获抽象基类"""
    
    @abstractmethod
    def Capture(self) -> Optional[np.ndarray]:
        """
        捕获屏幕
        
        Returns:
            捕获的帧（BGR格式），失败返回 None
        """
        pass
    
    @abstractmethod
    def GetScreenSize(self) -> tuple[int, int]:
        """
        获取屏幕尺寸
        
        Returns:
            (width, height)
        """
        pass


class MssScreenCapture(ScreenCapture):
    """使用 mss 实现的屏幕捕获"""
    
    def __init__(self, monitor_index: int = 1):
        """
        初始化屏幕捕获
        
        Args:
            monitor_index: 显示器索引（0=所有显示器，1=主显示器）
        """
        try:
            import mss
            self.sct_ = mss.mss()
            self.monitor_index_ = monitor_index
            self.monitor_ = self.sct_.monitors[monitor_index]
            self.width_ = self.monitor_['width']
            self.height_ = self.monitor_['height']
            logger.info(f"屏幕捕获初始化: {self.width_}x{self.height_}")
        except ImportError:
            logger.error("mss 未安装，请运行: pip install mss")
            raise
        except Exception as e:
            logger.error(f"屏幕捕获初始化失败: {e}")
            raise
    
    def Capture(self) -> Optional[np.ndarray]:
        """捕获屏幕"""
        try:
            frame_array = np.array(self.sct_.grab(self.monitor_))
            # 转换 BGRA 到 BGR
            frame_bgr = frame_array[:, :, :3]
            return frame_bgr
        except Exception as e:
            logger.error(f"屏幕捕获失败: {e}")
            return None
    
    def GetScreenSize(self) -> tuple[int, int]:
        """获取屏幕尺寸"""
        return (self.width_, self.height_)

