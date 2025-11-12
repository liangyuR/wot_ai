#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建标准化目录结构
用于初始化项目的标准数据存储目录
"""

from pathlib import Path
from .utils.paths import (
    get_datasets_dir,
    get_models_dir,
    get_training_dir,
    get_configs_dir,
    get_logs_dir,
    get_cache_dir,
    ensure_dir
)


def SetupDirectories() -> bool:
    """
    创建所有标准目录结构
    
    Returns:
        是否成功
    """
    try:
        # 数据集目录
        datasets_dir = get_datasets_dir()
        ensure_dir(datasets_dir / 'raw' / 'minimap')
        ensure_dir(datasets_dir / 'processed' / 'minimap')
        ensure_dir(datasets_dir / 'annotations')
        
        # 模型目录
        models_dir = get_models_dir()
        ensure_dir(models_dir / 'yolo' / 'minimap')
        ensure_dir(get_training_dir() / 'minimap')
        
        # 配置目录
        configs_dir = get_configs_dir()
        ensure_dir(configs_dir / 'data_collection')
        ensure_dir(configs_dir / 'yolo')
        ensure_dir(configs_dir / 'path_planning')
        
        # 日志目录
        ensure_dir(get_logs_dir())
        
        # 缓存目录
        ensure_dir(get_cache_dir())
        
        print("✓ 标准目录结构创建成功")
        return True
    except Exception as e:
        print(f"✗ 创建目录结构失败: {e}")
        return False


if __name__ == "__main__":
    SetupDirectories()

