#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径管理工具
提供统一的路径获取和 Python 路径设置功能
"""

import sys
from pathlib import Path
from typing import Optional, Union


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
    candidate = current_file.parent.parent
    
    # 检查是否是项目根目录（包含 pyproject.toml、requirements.txt 或 .git 等标识）
    if (candidate / "pyproject.toml").exists() or (candidate / "requirements.txt").exists() or (candidate / ".git").exists():
        _project_root = candidate
        return _project_root
    
    # 方法2: 从 sys.path 中查找
    for path_str in sys.path:
        if not path_str:
            continue
        path = Path(path_str).resolve()
        if (path / "pyproject.toml").exists() or (path / "requirements.txt").exists() or (path / ".git").exists():
            _project_root = path
            return _project_root
    
    # 方法3: 如果都找不到，使用当前文件的上两级目录作为默认值
    _project_root = current_file.parent.parent
    return _project_root


def get_data_dir() -> Path:
    """
    获取数据目录
    
    Returns:
        数据目录的 Path 对象
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    return data_dir


def get_config_dir() -> Path:
    """
    获取配置目录
    
    Returns:
        配置目录的 Path 对象
    """
    project_root = get_project_root()
    config_dir = project_root / "wot_ai" / "data_collection" / "configs"
    return config_dir


def get_model_dir() -> Path:
    """
    获取模型目录（旧版，保持向后兼容）
    
    Returns:
        模型目录的 Path 对象
    """
    return get_models_dir() / "yolo"


def get_datasets_dir() -> Path:
    """
    获取数据集目录
    
    Returns:
        数据集目录的 Path 对象
    """
    project_root = get_project_root()
    datasets_dir = project_root / "data" / "datasets"
    return datasets_dir


def get_models_dir() -> Path:
    """
    获取模型目录（统一）
    
    Returns:
        模型目录的 Path 对象
    """
    project_root = get_project_root()
    models_dir = project_root / "data" / "models"
    return models_dir


def get_training_dir() -> Path:
    """
    获取训练输出目录
    
    Returns:
        训练输出目录的 Path 对象
    """
    project_root = get_project_root()
    training_dir = project_root / "data" / "models" / "training"
    return training_dir


def get_configs_dir() -> Path:
    """
    获取统一配置目录
    
    Returns:
        配置目录的 Path 对象
    """
    project_root = get_project_root()
    configs_dir = project_root / "configs"
    return configs_dir


def get_logs_dir() -> Path:
    """
    获取日志目录
    
    Returns:
        日志目录的 Path 对象
    """
    project_root = get_project_root()
    logs_dir = project_root / "logs"
    return logs_dir


def get_cache_dir() -> Path:
    """
    获取缓存目录
    
    Returns:
        缓存目录的 Path 对象
    """
    project_root = get_project_root()
    cache_dir = project_root / "data" / "cache"
    return cache_dir


def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    解析路径（支持相对路径和绝对路径）
    
    Args:
        path: 路径字符串或 Path 对象
        base_dir: 基础目录，如果为 None 则使用项目根目录
    
    Returns:
        解析后的绝对路径
    """
    path_obj = Path(path)
    
    # 如果是绝对路径，直接返回
    if path_obj.is_absolute():
        return path_obj.resolve()
    
    # 如果是相对路径，基于 base_dir 解析
    if base_dir is None:
        base_dir = get_project_root()
    else:
        base_dir = Path(base_dir).resolve()
    
    return (base_dir / path_obj).resolve()


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        目录路径
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


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

