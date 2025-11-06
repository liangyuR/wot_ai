#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入辅助工具
提供统一的导入处理函数
"""

import sys
from typing import Any, Optional, Tuple
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

