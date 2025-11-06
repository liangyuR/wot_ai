#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径管理工具
提供统一的路径获取和 Python 路径设置功能
"""

import sys
from pathlib import Path
from typing import Optional


# 缓存项目根目录，避免重复计算
_project_root: Optional[Path] = None
_python_path_setup: bool = False


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录的 Path 对象
    """
    global _project_root
    
    if _project_root is not None:
        return _project_root
    
    # 尝试多种方式找到项目根目录
    # 方法1: 从当前文件位置向上查找（最可靠）
    current_file = Path(__file__).resolve()
    # wot_ai/utils/paths.py -> wot_ai -> 项目根目录
    candidate = current_file.parent.parent.parent
    
    # 检查是否是项目根目录（包含 requirements.txt 或 .git 等标识）
    if (candidate / "requirements.txt").exists() or (candidate / ".git").exists():
        _project_root = candidate
        return _project_root
    
    # 方法2: 从 sys.path 中查找
    for path_str in sys.path:
        if not path_str:
            continue
        path = Path(path_str).resolve()
        if (path / "requirements.txt").exists() or (path / ".git").exists():
            _project_root = path
            return _project_root
    
    # 方法3: 如果都找不到，使用当前文件的上两级目录作为默认值
    _project_root = current_file.parent.parent.parent
    return _project_root


def get_data_dir() -> Path:
    """
    获取数据目录
    
    Returns:
        数据目录的 Path 对象
    """
    project_root = get_project_root()
    data_dir = project_root / "data_collection" / "data"
    return data_dir


def get_config_dir() -> Path:
    """
    获取配置目录
    
    Returns:
        配置目录的 Path 对象
    """
    project_root = get_project_root()
    config_dir = project_root / "data_collection" / "configs"
    return config_dir


def get_model_dir() -> Path:
    """
    获取模型目录
    
    Returns:
        模型目录的 Path 对象
    """
    project_root = get_project_root()
    model_dir = project_root / "yolo" / "model"
    return model_dir


def setup_python_path(project_root: Optional[Path] = None) -> None:
    """
    自动设置 Python 路径，将项目根目录添加到 sys.path
    
    Args:
        project_root: 项目根目录，如果为 None 则自动查找
    """
    global _python_path_setup
    
    if _python_path_setup:
        return
    
    if project_root is None:
        project_root = get_project_root()
    else:
        project_root = Path(project_root).resolve()
    
    project_root_str = str(project_root)
    
    # 如果项目根目录不在 sys.path 中，则添加
    if project_root_str not in sys.path:
        # 插入到最前面，优先使用项目内的模块
        sys.path.insert(0, project_root_str)
    
    _python_path_setup = True

