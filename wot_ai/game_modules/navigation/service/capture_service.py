#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
屏幕截取服务：封装屏幕捕获逻辑
"""

from typing import Optional
import numpy as np
import logging

# 统一导入机制
from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple
setup_python_path()

SetupLogger = None
logger_module, _ = try_import_multiple([
    'wot_ai.game_modules.common.utils.logger',
    'game_modules.common.utils.logger',
    'common.utils.logger',
    'yolo.utils.logger'
])
if logger_module is not None:
    SetupLogger = getattr(logger_module, 'SetupLogger', None)

if SetupLogger is None:
    from ..common.utils.logger import SetupLogger

logger = SetupLogger(__name__)


class CaptureService:
    """屏幕截取服务"""
    
    def __init__(self, monitor_index: int = 1):
        """
        初始化屏幕截取服务
        
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
            logger.info(f"屏幕截取服务初始化: {self.width_}x{self.height_}")
        except ImportError:
            logger.error("mss 未安装，请运行: pip install mss")
            raise
        except Exception as e:
            logger.error(f"屏幕截取服务初始化失败: {e}")
            raise
    
    def Capture(self) -> Optional[np.ndarray]:
        """
        捕获屏幕
        
        Returns:
            捕获的帧（BGR格式），失败返回 None
        """
        try:
            frame_array = np.array(self.sct_.grab(self.monitor_))
            # 转换 BGRA 到 BGR
            frame_bgr = frame_array[:, :, :3]
            return frame_bgr
        except Exception as e:
            logger.error(f"屏幕捕获失败: {e}")
            return None
    
    def GetScreenSize(self) -> tuple[int, int]:
        """
        获取屏幕尺寸
        
        Returns:
            (width, height)
        """
        return (self.width_, self.height_)

