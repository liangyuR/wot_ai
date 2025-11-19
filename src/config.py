#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置模块
定义项目级别的配置常量和路径配置
"""

import sys
from pathlib import Path
from typing import Dict, Any
from .utils.paths import (
    get_project_root,
    get_datasets_dir,
    get_models_dir,
    get_training_dir,
    get_configs_dir,
    get_logs_dir,
    get_cache_dir
)


# =============================
# 路径配置
# =============================

def GetProjectPaths() -> Dict[str, Path]:
    """
    获取项目路径配置
    
    Returns:
        路径配置字典
    """
    return {
        'project_root': get_project_root(),
        'datasets_dir': get_datasets_dir(),
        'models_dir': get_models_dir(),
        'training_dir': get_training_dir(),
        'configs_dir': get_configs_dir(),
        'logs_dir': get_logs_dir(),
        'cache_dir': get_cache_dir(),
    }


# =============================
# 数据集路径配置
# =============================

def GetDatasetPaths() -> Dict[str, Path]:
    """
    获取数据集相关路径
    
    Returns:
        数据集路径配置字典
    """
    datasets_dir = get_datasets_dir()
    return {
        'raw': datasets_dir / 'raw',
        'processed': datasets_dir / 'processed',
        'annotations': datasets_dir / 'annotations',
    }


# =============================
# 模型路径配置
# =============================

def GetModelPaths() -> Dict[str, Path]:
    """
    获取模型相关路径
    
    Returns:
        模型路径配置字典
    """
    models_dir = get_models_dir()
    return {
        'yolo_minimap': models_dir / 'yolo' / 'minimap',
        'training': get_training_dir(),
    }


# =============================
# 配置路径配置
# =============================

def GetConfigPaths() -> Dict[str, Path]:
    """
    获取配置文件路径
    
    Returns:
        配置路径配置字典
    """
    configs_dir = get_configs_dir()
    return {
        'data_collection': configs_dir / 'data_collection',
        'yolo': configs_dir / 'yolo',
        'path_planning': configs_dir / 'path_planning',
    }


# =============================
# 项目常量
# =============================

# 项目版本
PROJECT_VERSION = "0.1.0"

# 项目名称
PROJECT_NAME = "wot_ai"

# 默认配置
DEFAULT_CONFIG = {
    'paths': GetProjectPaths(),
    'datasets': GetDatasetPaths(),
    'models': GetModelPaths(),
    'configs': GetConfigPaths(),
}


def GetConfig() -> Dict[str, Any]:
    """
    获取完整配置
    
    Returns:
        完整配置字典
    """
    return DEFAULT_CONFIG.copy()


def get_program_dir() -> Path:
    """
    获取程序运行目录（优先 game_modules，打包后返回 EXE 目录）
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    
    module_dir = Path(__file__).resolve().parent
    candidate = module_dir / "game_modules"
    if candidate.exists():
        return candidate
    return module_dir

