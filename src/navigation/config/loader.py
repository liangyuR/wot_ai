#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器

从YAML文件加载配置并使用Pydantic验证。
"""

import yaml
from pathlib import Path
from typing import Optional, Any, Dict
from loguru import logger
from pydantic import ValidationError

from src.navigation.config.models import NavigationConfig


def load_config(config_path: Path, base_dir: Optional[Path] = None) -> NavigationConfig:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
        base_dir: 用于解析相对路径的程序目录（main_window.py 或 EXE 所在目录）
    
    Returns:
        验证后的NavigationConfig对象
    
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML格式错误
        ValidationError: 配置验证失败
    """
    config_path = Path(config_path)
    if base_dir is None:
        default_base = config_path.resolve().parent
        if default_base.name.lower() == "config":
            default_base = default_base.parent
        base_dir = default_base
    else:
        base_dir = Path(base_dir).resolve()
    
    # 检查文件是否存在
    if not config_path.exists():
        error_msg = f"配置文件不存在: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # 加载YAML文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        error_msg = f"YAML格式错误: {e}"
        logger.error(error_msg)
        raise yaml.YAMLError(error_msg) from e
    except Exception as e:
        error_msg = f"读取配置文件失败: {e}"
        logger.error(error_msg)
        raise IOError(error_msg) from e
    
    if raw_config is None:
        error_msg = "配置文件为空"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 解析相对路径
    if isinstance(raw_config, dict):
        _apply_relative_paths(raw_config, base_dir)
    
    # 使用Pydantic验证配置
    try:
        config = NavigationConfig(**raw_config)
        logger.info(f"配置加载成功: {config_path}")
        return config
    except ValidationError as e:
        error_msg = f"配置验证失败:\n{e}"
        logger.error(error_msg)
        # 输出详细的验证错误信息
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            logger.error(f"  {field_path}: {error['msg']}")
        raise ValidationError(error_msg) from e


def _resolve_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    """解析相对路径为绝对路径"""
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _apply_relative_paths(raw_config: Dict[str, Any], base_dir: Path) -> None:
    """将配置中的相对路径字段转换为绝对路径"""
    model_cfg = raw_config.get('model')
    if isinstance(model_cfg, dict) and 'path' in model_cfg and model_cfg['path']:
        model_cfg['path'] = _resolve_path(model_cfg['path'], base_dir)
    
    mask_cfg = raw_config.get('mask')
    if isinstance(mask_cfg, dict):
        if mask_cfg.get('path'):
            mask_cfg['path'] = _resolve_path(mask_cfg['path'], base_dir)
        if mask_cfg.get('directory'):
            mask_cfg['directory'] = _resolve_path(mask_cfg['directory'], base_dir)
    
    vehicle_cfg = raw_config.get('vehicle')
    if isinstance(vehicle_cfg, dict) and vehicle_cfg.get('screenshot_dir'):
        vehicle_cfg['screenshot_dir'] = _resolve_path(vehicle_cfg['screenshot_dir'], base_dir)

