#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入辅助工具
提供统一的导入处理函数和类
"""

import sys
import functools
from typing import Any, Optional, Tuple, List, Callable
from pathlib import Path

from .paths import setup_python_path, get_project_root


def safe_import(module_name: str, package: Optional[str] = None) -> Tuple[Any, bool]:
    """
    安全导入模块，如果失败则尝试设置路径后重新导入
    
    Args:
        module_name: 模块名称，如 'yolo.core.cv_minimap_detector'
        package: 包名称，用于相对导入（可选）
    
    Returns:
        (module, success): 模块对象和是否成功导入的布尔值
    """
    # 首先尝试直接导入
    try:
        if package:
            module = __import__(module_name, fromlist=[package])
        else:
            module = __import__(module_name)
        return module, True
    except ImportError:
        pass
    
    # 如果失败，尝试设置路径后重新导入
    try:
        setup_python_path()
        if package:
            module = __import__(module_name, fromlist=[package])
        else:
            module = __import__(module_name)
        return module, True
    except ImportError as e:
        return None, False


def import_from_module(module_name: str, item_name: str) -> Tuple[Any, bool]:
    """
    从模块中导入特定对象
    
    Args:
        module_name: 模块名称，如 'yolo.core.cv_minimap_detector'
        item_name: 要导入的对象名称，如 'CvMinimapDetector'
    
    Returns:
        (item, success): 导入的对象和是否成功导入的布尔值
    """
    module, success = safe_import(module_name)
    if not success:
        return None, False
    
    try:
        item = getattr(module, item_name)
        return item, True
    except AttributeError:
        return None, False


class ImportHelper:
    """
    统一的导入辅助类
    提供多种导入方式和自动路径设置
    """
    
    def __init__(self, auto_setup_path: bool = True):
        """
        初始化导入辅助器
        
        Args:
            auto_setup_path: 是否自动设置 Python 路径
        """
        self.auto_setup_path_ = auto_setup_path
        if auto_setup_path:
            setup_python_path()
    
    def import_module(self, module_name: str, package: Optional[str] = None) -> Any:
        """
        导入模块
        
        Args:
            module_name: 模块名称
            package: 包名称（用于相对导入）
        
        Returns:
            模块对象
        
        Raises:
            ImportError: 如果导入失败
        """
        module, success = safe_import(module_name, package)
        if not success:
            raise ImportError(f"无法导入模块: {module_name}")
        return module
    
    def import_from(self, module_name: str, item_name: str) -> Any:
        """
        从模块中导入特定对象
        
        Args:
            module_name: 模块名称
            item_name: 对象名称
        
        Returns:
            导入的对象
        
        Raises:
            ImportError: 如果导入失败
            AttributeError: 如果对象不存在
        """
        item, success = import_from_module(module_name, item_name)
        if not success:
            raise ImportError(f"无法从 {module_name} 导入 {item_name}")
        return item
    
    def try_import(self, module_name: str, default: Any = None) -> Any:
        """
        尝试导入模块，失败则返回默认值
        
        Args:
            module_name: 模块名称
            default: 默认值
        
        Returns:
            模块对象或默认值
        """
        module, success = safe_import(module_name)
        return module if success else default
    
    def try_import_from(self, module_name: str, item_name: str, default: Any = None) -> Any:
        """
        尝试从模块导入对象，失败则返回默认值
        
        Args:
            module_name: 模块名称
            item_name: 对象名称
            default: 默认值
        
        Returns:
            对象或默认值
        """
        item, success = import_from_module(module_name, item_name)
        return item if success else default


def with_path_setup(func: Callable) -> Callable:
    """
    装饰器：自动设置 Python 路径
    
    Args:
        func: 被装饰的函数
    
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        setup_python_path()
        return func(*args, **kwargs)
    return wrapper


def try_import_multiple(module_names: List[str], package: Optional[str] = None) -> Tuple[Any, Optional[str]]:
    """
    尝试从多个模块名导入，返回第一个成功的
    
    Args:
        module_names: 模块名称列表
        package: 包名称（用于相对导入）
    
    Returns:
        (module, module_name): 模块对象和成功的模块名，如果都失败则返回 (None, None)
    """
    for module_name in module_names:
        module, success = safe_import(module_name, package)
        if success:
            return module, module_name
    return None, None


def import_function(module_names: List[str], function_name: str, package: Optional[str] = None) -> Optional[Any]:
    """
    从多个可能的模块中导入指定函数（现代化便捷方法）
    
    Args:
        module_names: 模块名列表（按优先级排序）
        function_name: 要导入的函数名
        package: 包名称（用于相对导入）
    
    Returns:
        函数对象，如果都失败则返回 None
    
    Example:
        SetupLogger = import_function([
            'wot_ai.game_modules.common.utils.logger',
            'game_modules.common.utils.logger',
            'common.utils.logger'
        ], 'SetupLogger')
        if SetupLogger is None:
            from ..common.utils.logger import SetupLogger
    """
    for module_name in module_names:
        module, success = safe_import(module_name, package)
        if success:
            func = getattr(module, function_name, None)
            if func is not None:
                return func
    return None

