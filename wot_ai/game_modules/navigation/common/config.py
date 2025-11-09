#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块：统一管理默认配置值，提供配置验证机制
"""

from typing import Dict, Any, Optional
from pathlib import Path
from ..common.constants import (
    DEFAULT_MINIMAP_WIDTH,
    DEFAULT_MINIMAP_HEIGHT,
    DEFAULT_GRID_SIZE,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MOUSE_SENSITIVITY,
    DEFAULT_CALIBRATION_FACTOR,
    DEFAULT_MOVE_SPEED,
    DEFAULT_ROTATION_SMOOTH,
    SMOOTH_WEIGHT_DEFAULT
)


class NavigationConfig:
    """导航模块配置管理器"""
    
    @staticmethod
    def GetDefaultConfig() -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            'minimap': {
                'region': {
                    'x': 1600,
                    'y': 800,
                    'width': DEFAULT_MINIMAP_WIDTH,
                    'height': DEFAULT_MINIMAP_HEIGHT
                },
                'grid_size': list(DEFAULT_GRID_SIZE),
                'scale_factor': DEFAULT_SCALE_FACTOR
            },
            'model': {
                'path': 'train/model/minimap_yolo.pt',
                'classes': 6,
                'conf_threshold': DEFAULT_CONFIDENCE_THRESHOLD
            },
            'navigation': {
                'mouse_sensitivity': DEFAULT_MOUSE_SENSITIVITY,
                'calibration_factor': DEFAULT_CALIBRATION_FACTOR,
                'move_speed': DEFAULT_MOVE_SPEED,
                'rotation_smooth': DEFAULT_ROTATION_SMOOTH
            },
            'planner': {
                'enable_smoothing': True,
                'smooth_weight': SMOOTH_WEIGHT_DEFAULT
            },
            'logging': {
                'level': 'INFO',
                'save_video': False
            }
        }
    
    @staticmethod
    def ValidateConfig(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        验证配置的有效性
        
        Args:
            config: 配置字典
        
        Returns:
            (是否有效, 错误信息)
        """
        # 验证minimap配置
        if 'minimap' in config:
            minimap = config['minimap']
            if 'region' in minimap:
                region = minimap['region']
                required_keys = ['x', 'y', 'width', 'height']
                for key in required_keys:
                    if key not in region:
                        return False, f"minimap.region缺少必需字段: {key}"
                    if not isinstance(region[key], int) or region[key] < 0:
                        return False, f"minimap.region.{key}必须是正整数"
        
        # 验证model配置
        if 'model' in config:
            model = config['model']
            if 'conf_threshold' in model:
                conf = model['conf_threshold']
                if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                    return False, "model.conf_threshold必须在0-1之间"
        
        # 验证navigation配置
        if 'navigation' in config:
            nav = config['navigation']
            if 'mouse_sensitivity' in nav:
                sens = nav['mouse_sensitivity']
                if not isinstance(sens, (int, float)) or sens <= 0:
                    return False, "navigation.mouse_sensitivity必须大于0"
        
        # 验证planner配置
        if 'planner' in config:
            planner = config['planner']
            if 'smooth_weight' in planner:
                weight = planner['smooth_weight']
                if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
                    return False, "planner.smooth_weight必须在0-1之间"
        
        return True, None
    
    @staticmethod
    def MergeConfig(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并配置字典
        
        Args:
            default: 默认配置
            override: 覆盖配置
        
        Returns:
            合并后的配置
        """
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = NavigationConfig.MergeConfig(result[key], value)
            else:
                result[key] = value
        
        return result

