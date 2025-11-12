#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能档位配置模块：管理不同性能模式下的参数
"""

# 标准库导入
from typing import Dict

# 本地模块导入
from ..common.imports import GetLogger
from ..common.constants import (
    FPS_HIGH, FPS_MEDIUM, FPS_LOW,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW,
    SCALE_HIGH, SCALE_MEDIUM, SCALE_LOW
)

logger = GetLogger()(__name__)


class PerformanceProfile:
    """性能档位配置管理器"""
    
    MODES = {
        "high": {
            "fps": FPS_HIGH,
            "conf": CONFIDENCE_HIGH,
            "scale": SCALE_HIGH
        },
        "medium": {
            "fps": FPS_MEDIUM,
            "conf": CONFIDENCE_MEDIUM,
            "scale": SCALE_MEDIUM
        },
        "low": {
            "fps": FPS_LOW,
            "conf": CONFIDENCE_LOW,
            "scale": SCALE_LOW
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
    
    def SetMode(self, mode: str) -> None:
        """
        设置性能模式
        
        Args:
            mode: 性能模式 ("high", "medium", "low")
        
        Raises:
            ValueError: 无效的性能模式
        """
        if not isinstance(mode, str):
            raise ValueError("mode必须是字符串")
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

