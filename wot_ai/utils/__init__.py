#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块
提供路径管理和导入辅助功能
"""

from .paths import (
    get_project_root,
    get_data_dir,
    get_config_dir,
    get_model_dir,
    setup_python_path
)

__all__ = [
    'get_project_root',
    'get_data_dir',
    'get_config_dir',
    'get_model_dir',
    'setup_python_path',
]

