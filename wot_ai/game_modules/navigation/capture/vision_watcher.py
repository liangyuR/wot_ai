#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时屏幕捕获模块：使用 mss 捕获游戏画面

职责：用于实时监控场景，捕获指定屏幕区域，支持按帧率流式输出
使用场景：NavigationMonitor等需要持续监控特定区域的场景
"""

# 标准库导入
from typing import Dict, Generator, Optional
import time

# 第三方库导入
import cv2
import numpy as np

# 本地模块导入
from ..common.exceptions import CaptureError
from loguru import logger

class VisionWatcher:
    """
    实时屏幕捕获器：按指定帧率捕获屏幕区域
    
    职责：用于实时监控场景，捕获指定屏幕区域，支持流式输出
    与CaptureService的区别：
    - VisionWatcher：捕获指定区域，支持流式输出，适合实时监控
    - CaptureService：捕获整个屏幕，提供屏幕尺寸信息，适合全屏捕获
    """
    
    def __init__(self, region: Dict, target_fps: float = 15.0):
        """
        初始化屏幕捕获器
        
        Args:
            region: 捕获区域配置 {'x': int, 'y': int, 'width': int, 'height': int}
            target_fps: 目标帧率
        
        Raises:
            ValueError: 输入参数无效
        """
        if not region or not isinstance(region, dict):
            raise ValueError("region必须是有效的字典")
        required_keys = ['x', 'y', 'width', 'height']
        for key in required_keys:
            if key not in region:
                raise ValueError(f"region缺少必需字段: {key}")
        if not isinstance(target_fps, (int, float)) or target_fps <= 0:
            raise ValueError("target_fps必须是正数")
        
        self.region_ = region
        self.target_fps_ = target_fps
        self.interval_ = 1.0 / target_fps if target_fps > 0 else 0.033
        self.sct_ = None
        self.InitMss_()
    
    def InitMss_(self):
        """初始化 mss 捕获器"""
        try:
            import mss
            self.sct_ = mss.mss()
            logger.info(f"屏幕捕获器初始化成功，目标帧率: {self.target_fps_} FPS")
        except ImportError as e:
            error_msg = "mss 未安装，请运行: pip install mss"
            logger.error(error_msg)
            raise CaptureError(error_msg) from e
        except Exception as e:
            error_msg = f"屏幕捕获器初始化失败: {e}"
            logger.error(error_msg)
            raise CaptureError(error_msg) from e
    
    def Capture(self) -> Optional[np.ndarray]:
        """
        捕获一帧屏幕
        
        Returns:
            BGR 格式的 numpy 数组，失败返回 None
        """
        if self.sct_ is None:
            return None
        
        try:
            # mss 需要 'top', 'left', 'width', 'height' 格式
            monitor = {
                'top': self.region_['y'],
                'left': self.region_['x'],
                'width': self.region_['width'],
                'height': self.region_['height']
            }
            
            screenshot = self.sct_.grab(monitor)
            img = np.array(screenshot)
            
            # mss 返回 BGRA 格式，转换为 BGR
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
        except Exception as e:
            error_msg = f"屏幕捕获失败: {e}"
            logger.error(error_msg)
            # 捕获失败不抛出异常，返回None以便调用者处理
            return None
    
    def Stream(self) -> Generator[np.ndarray, None, None]:
        """
        生成器：持续捕获屏幕帧
        
        Yields:
            BGR 格式的 numpy 数组
        """
        while True:
            frame = self.Capture()
            if frame is not None:
                yield frame
            time.sleep(self.interval_)
    
    def SetFps(self, fps: float) -> None:
        """
        设置目标帧率
        
        Args:
            fps: 目标帧率
        
        Raises:
            ValueError: 无效的帧率值
        """
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError("fps必须是正数")
        
        self.target_fps_ = fps
        self.interval_ = 1.0 / fps if fps > 0 else 0.033
        logger.info(f"更新目标帧率: {fps} FPS")

