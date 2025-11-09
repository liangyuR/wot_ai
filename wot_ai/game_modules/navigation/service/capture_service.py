#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
屏幕截取服务：封装屏幕捕获逻辑

职责：用于全屏捕获场景，捕获整个屏幕，提供屏幕尺寸信息
使用场景：PathPlanningController等需要全屏捕获的场景
"""

# 标准库导入
from typing import Optional

# 第三方库导入
import cv2
import numpy as np

# 本地模块导入
from ..common.imports import GetLogger
from ..common.exceptions import CaptureError, InitializationError

logger = GetLogger()(__name__)


class CaptureService:
    """
    屏幕截取服务
    
    职责：用于全屏捕获场景，捕获整个屏幕，提供屏幕尺寸信息
    与VisionWatcher的区别：
    - CaptureService：捕获整个屏幕，提供屏幕尺寸信息，适合全屏捕获
    - VisionWatcher：捕获指定区域，支持流式输出，适合实时监控
    """
    
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
        except ImportError as e:
            error_msg = "mss 未安装，请运行: pip install mss"
            logger.error(error_msg)
            raise InitializationError(error_msg) from e
        except (IndexError, KeyError) as e:
            error_msg = f"无效的显示器索引: {monitor_index}"
            logger.error(error_msg)
            raise InitializationError(error_msg) from e
        except Exception as e:
            error_msg = f"屏幕截取服务初始化失败: {e}"
            logger.error(error_msg)
            raise InitializationError(error_msg) from e
    
    def Capture(self) -> Optional[np.ndarray]:
        """
        捕获屏幕
        
        Returns:
            捕获的帧（BGR格式），失败返回 None
        """
        try:
            frame_array = np.array(self.sct_.grab(self.monitor_))
            # 转换 BGRA 到 BGR
            if frame_array.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame_array[:, :, :3]
            return frame_bgr
        except Exception as e:
            error_msg = f"屏幕捕获失败: {e}"
            logger.error(error_msg)
            # 捕获失败不抛出异常，返回None以便调用者处理
            return None
    
    def GetScreenSize(self) -> tuple[int, int]:
        """
        获取屏幕尺寸
        
        Returns:
            (width, height)
        """
        return (self.width_, self.height_)

