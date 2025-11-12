#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载工具
统一从配置文件加载类别定义和其他配置
"""

from pathlib import Path
from typing import List, Optional
import yaml
from wot_ai.utils.paths import get_project_root


def LoadClassesFromConfig(config_path: Optional[Path] = None) -> List[str]:
    """
    从配置文件加载类别定义
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认路径
    
    Returns:
        类别名称列表
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "wot_ai" / "train_yolo" / "train_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        classes = config.get('classes', [])
        if not classes:
            raise ValueError(f"配置文件中未找到类别定义: {config_path}")
        
        # 如果类别是字符串列表，直接返回
        if isinstance(classes, list):
            # 移除注释（如果有）
            clean_classes = []
            for cls in classes:
                if isinstance(cls, str):
                    # 移除行内注释
                    cls = cls.split('#')[0].strip()
                    if cls:
                        clean_classes.append(cls)
                else:
                    clean_classes.append(str(cls))
            return clean_classes
        
        raise ValueError(f"配置文件中的类别定义格式不正确: {classes}")
    except Exception as e:
        raise RuntimeError(f"加载类别定义失败: {e}")


def GetNumClasses(config_path: Optional[Path] = None) -> int:
    """
    获取类别数量
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认路径
    
    Returns:
        类别数量
    """
    classes = LoadClassesFromConfig(config_path)
    return len(classes)

