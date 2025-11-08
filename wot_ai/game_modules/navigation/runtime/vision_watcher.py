#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时屏幕捕获模块：使用 mss 捕获游戏画面
"""

import time
from typing import Dict, Generator, Optional
import numpy as np
import cv2

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
    from ...common.utils.logger import SetupLogger

logger = SetupLogger(__name__)


class VisionWatcher:
    """实时屏幕捕获器：按指定帧率捕获屏幕区域"""
    
    def __init__(self, region: Dict, target_fps: float = 15.0):
        """
        初始化屏幕捕获器
        
        Args:
            region: 捕获区域配置 {'x': int, 'y': int, 'width': int, 'height': int}
            target_fps: 目标帧率
        """
        self.region_ = region
        self.target_fps_ = target_fps
        self.interval_ = 1.0 / target_fps if target_fps > 0 else 0.033
        self.sct_ = None
        self._InitMss()
    
    def _InitMss(self):
        """初始化 mss 捕获器"""
        try:
            import mss
            self.sct_ = mss.mss()
            logger.info(f"屏幕捕获器初始化成功，目标帧率: {self.target_fps_} FPS")
        except ImportError:
            logger.error("mss 未安装，请运行: pip install mss")
            raise
        except Exception as e:
            logger.error(f"屏幕捕获器初始化失败: {e}")
            raise
    
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
            logger.error(f"屏幕捕获失败: {e}")
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
    
    def SetFps(self, fps: float):
        """
        设置目标帧率
        
        Args:
            fps: 目标帧率
        """
        self.target_fps_ = fps
        self.interval_ = 1.0 / fps if fps > 0 else 0.033
        logger.info(f"更新目标帧率: {fps} FPS")

