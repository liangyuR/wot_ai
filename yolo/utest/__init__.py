#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试包初始化
自动设置 Python 路径，确保测试文件可以正确导入项目模块
"""

from pathlib import Path
import sys

# 尝试使用 wot_ai 包的路径设置功能
try:
    from wot_ai.utils.paths import setup_python_path
    setup_python_path()
except ImportError:
    # 如果 wot_ai 包不可用，手动设置路径
    # 从当前文件位置找到项目根目录
    # yolo/utest/__init__.py -> yolo -> 项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

