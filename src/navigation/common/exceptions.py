#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义异常类：定义导航模块的专用异常
"""


class NavigationError(Exception):
    """导航模块基础异常类"""
    pass


class ModelLoadError(NavigationError):
    """模型加载失败异常"""
    pass


class DetectionError(NavigationError):
    """检测失败异常"""
    pass


class PathPlanningError(NavigationError):
    """路径规划失败异常"""
    pass


class ConfigurationError(NavigationError):
    """配置错误异常"""
    pass


class InitializationError(NavigationError):
    """初始化失败异常"""
    pass


class CaptureError(NavigationError):
    """屏幕捕获失败异常"""
    pass


class ControlError(NavigationError):
    """控制操作失败异常"""
    pass

