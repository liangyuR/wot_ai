#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
屏幕截取服务：封装屏幕捕获逻辑

职责：用于全屏捕获场景，捕获整个屏幕，提供屏幕尺寸信息
使用场景：PathPlanningController等需要全屏捕获的场景
"""

# 标准库导入
from typing import Optional, Dict
import threading

# 第三方库导入
import cv2
import numpy as np

# 本地模块导入
from loguru import logger


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
            self.monitor_index_ = monitor_index
            # 使用线程本地存储，每个线程都有自己的 mss 实例
            self.thread_local_ = threading.local()
            self.mss_module_ = mss
            
            # 在主线程中初始化一次，获取显示器信息
            temp_sct = mss.mss()
            self.monitor_ = temp_sct.monitors[monitor_index]
            self.width_ = self.monitor_['width']
            self.height_ = self.monitor_['height']
            logger.info(f"屏幕截取服务初始化: {self.width_}x{self.height_}")
        except ImportError as e:
            error_msg = "mss 未安装，请运行: pip install mss"
            logger.error(error_msg)
        except (IndexError, KeyError) as e:
            error_msg = f"无效的显示器索引: {monitor_index}"
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"屏幕截取服务初始化失败: {e}"
            logger.error(error_msg)
    
    def _GetMssInstance_(self):
        """
        获取当前线程的 mss 实例（线程安全）
        
        Returns:
            当前线程的 mss.mss() 实例
        """
        if not hasattr(self.thread_local_, 'sct'):
            self.thread_local_.sct = self.mss_module_.mss()
        return self.thread_local_.sct
    
    def Capture(self, minimap_region: Optional[Dict[str, int]] = None) -> Optional[np.ndarray]:
        """
        捕获屏幕（线程安全）
        
        Args:
            minimap_region: 可选的小地图区域字典 {'x': int, 'y': int, 'width': int, 'height': int}，
                          如果为 None 则捕获整个屏幕
        
        Returns:
            捕获的帧（BGR格式），失败返回 None
        """
        try:
            # 获取当前线程的 mss 实例
            sct = self._GetMssInstance_()
            
            # 如果指定了区域，捕获指定区域；否则捕获整个屏幕
            if minimap_region is not None:
                mon = {
                    'left': minimap_region['x'],
                    'top': minimap_region['y'],
                    'width': minimap_region['width'],
                    'height': minimap_region['height']
                }
            else:
                # 捕获整个屏幕
                mon = self.monitor_
            
            frame_array = np.array(sct.grab(mon))
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
    
    def CaptureExcludingOverlay(self, overlay) -> Optional[np.ndarray]:
        """
        捕获屏幕，临时隐藏 overlay 窗口以避免干扰
        
        Args:
            overlay: TransparentOverlay 实例，如果为 None 则正常捕获
        
        Returns:
            捕获的帧（BGR格式），失败返回 None
        """
        was_visible = False
        
        # 如果 overlay 存在且可见，先隐藏它
        if overlay is not None:
            try:
                was_visible = overlay.IsVisible()
                if was_visible:
                    overlay.Hide()
            except Exception as e:
                logger.warning(f"隐藏 overlay 失败，继续捕获: {e}")
        
        try:
            # 执行正常的捕获操作
            frame = self.Capture()
            return frame
        finally:
            # 确保即使出错也能恢复 overlay 显示
            if overlay is not None and was_visible:
                try:
                    overlay.Show()
                except Exception as e:
                    logger.error(f"恢复 overlay 显示失败: {e}")
    
    def GetScreenSize(self) -> tuple[int, int]:
        """
        获取屏幕尺寸
        
        Returns:
            (width, height)
        """
        return (self.width_, self.height_)

