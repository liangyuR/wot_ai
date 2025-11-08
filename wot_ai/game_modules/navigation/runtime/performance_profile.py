#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能档位配置模块：管理不同性能模式下的参数
"""

from typing import Dict

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


class PerformanceProfile:
    """性能档位配置管理器"""
    
    MODES = {
        "high": {
            "fps": 30,
            "conf": 0.25,
            "scale": 1.0
        },
        "medium": {
            "fps": 15,
            "conf": 0.35,
            "scale": 0.75
        },
        "low": {
            "fps": 8,
            "conf": 0.45,
            "scale": 0.6
        }
    }
    
    def __init__(self, mode: str = "medium"):
        """
        初始化性能配置
        
        Args:
            mode: 性能模式 ("high", "medium", "low")
        """
        self.mode_ = mode
        self.fps_ = 15.0
        self.conf_ = 0.35
        self.scale_ = 0.75
        self.SetMode(mode)
    
    def SetMode(self, mode: str):
        """
        设置性能模式
        
        Args:
            mode: 性能模式 ("high", "medium", "low")
        """
        if mode not in self.MODES:
            logger.warning(f"未知的性能模式: {mode}，使用默认模式 medium")
            mode = "medium"
        
        profile = self.MODES[mode]
        self.mode_ = mode
        self.fps_ = float(profile["fps"])
        self.conf_ = float(profile["conf"])
        self.scale_ = float(profile["scale"])
        
        logger.info(f"性能模式设置为: {mode} (FPS: {self.fps_}, 置信度: {self.conf_}, 缩放: {self.scale_})")
    
    def GetFps(self) -> float:
        """
        获取目标帧率
        
        Returns:
            目标帧率
        """
        return self.fps_
    
    def GetConfidence(self) -> float:
        """
        获取 YOLO 置信度阈值
        
        Returns:
            置信度阈值
        """
        return self.conf_
    
    def GetScale(self) -> float:
        """
        获取显示缩放因子
        
        Returns:
            缩放因子
        """
        return self.scale_
    
    def GetMode(self) -> str:
        """
        获取当前性能模式
        
        Returns:
            性能模式名称
        """
        return self.mode_

