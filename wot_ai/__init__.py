#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WoT AI 主包
提供统一的路径管理和导入机制
"""

__version__ = "0.1.0"

# 延迟导入，避免循环导入问题
def _setup_path():
    """延迟设置路径"""
    try:
        from .utils.paths import setup_python_path
        setup_python_path()
    except ImportError:
        pass

# 自动设置 Python 路径（延迟执行）
_setup_path()

__all__ = []

