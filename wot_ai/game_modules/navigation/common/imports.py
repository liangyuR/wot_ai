#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一导入机制：简化模块导入逻辑
"""

from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple, import_function

# 初始化路径
setup_python_path()


def GetLogger():
    """
    获取统一的日志记录器
    
    Returns:
        SetupLogger函数
    """
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
    
    return SetupLogger


def ImportNavigationModule(module_name: str, class_name: str = None):
    """
    导入navigation模块中的类或模块
    
    Args:
        module_name: 模块名称（相对于navigation目录）
        class_name: 类名称（如果为None，返回模块）
    
    Returns:
        导入的类或模块，失败返回None
    """
    if class_name:
        return import_function([
            f'wot_ai.game_modules.navigation.{module_name}',
            f'game_modules.navigation.{module_name}',
            f'navigation.{module_name}'
        ], class_name)
    else:
        module, _ = try_import_multiple([
            f'wot_ai.game_modules.navigation.{module_name}',
            f'game_modules.navigation.{module_name}',
            f'navigation.{module_name}'
        ])
        return module

