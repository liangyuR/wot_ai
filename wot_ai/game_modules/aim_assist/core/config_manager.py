#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器：加载、保存、验证配置
"""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import logging

logger = logging.getLogger(__name__)


class AimConfigManager:
    """瞄准辅助配置管理器"""
    
    def __init__(self, config_path: Path):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path_ = Path(config_path)
        self.config_ = {}
    
    def Load(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            if not self.config_path_.exists():
                logger.warning(f"配置文件不存在: {self.config_path_}，使用默认配置")
                self.config_ = self.GetDefaultConfig()
                return self.config_
            
            with open(self.config_path_, 'r', encoding='utf-8') as f:
                self.config_ = yaml.safe_load(f) or {}
            
            # 合并默认配置（确保所有必需字段存在）
            default_config = self.GetDefaultConfig()
            self.config_ = self._MergeConfig(default_config, self.config_)
            
            logger.info(f"配置加载成功: {self.config_path_}")
            return self.config_
        except Exception as e:
            logger.error(f"加载配置失败: {e}，使用默认配置")
            self.config_ = self.GetDefaultConfig()
            return self.config_
    
    def Save(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存配置文件
        
        Args:
            config: 配置字典（如果None则使用当前配置）
        
        Returns:
            是否保存成功
        """
        try:
            if config is not None:
                self.config_ = config
            
            # 确保配置目录存在
            self.config_path_.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path_, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"配置已保存: {self.config_path_}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def Get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置键路径（如 'screen.width', 'fov.horizontal'）
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config_
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def Set(self, key_path: str, value: Any):
        """
        设置配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置键路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config_
        
        # 创建嵌套字典
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def GetDefaultConfig(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'screen': {
                'width': 1920,
                'height': 1080
            },
            'fov': {
                'horizontal': 90.0,
                'vertical': None
            },
            'mouse': {
                'sensitivity': 1.0,
                'calibration_factor': None
            },
            'smoothing': {
                'factor': 0.3,
                'max_step': 50.0
            },
            'hotkeys': {
                'toggle': 'f8',
                'exit': 'esc'
            },
            'detection': {
                'model_path': 'train/model/yolo11n.pt',
                'target_class': None,
                'fps': 30.0
            }
        }
    
    def _MergeConfig(self, default: Dict, override: Dict) -> Dict:
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
                result[key] = self._MergeConfig(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def Validate(self) -> tuple[bool, list[str]]:
        """
        验证配置
        
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 验证屏幕分辨率
        screen_w = self.Get('screen.width', 0)
        screen_h = self.Get('screen.height', 0)
        if screen_w <= 0 or screen_h <= 0:
            errors.append("屏幕分辨率无效")
        
        # 验证 FOV
        h_fov = self.Get('fov.horizontal', 0)
        if h_fov <= 0 or h_fov > 180:
            errors.append("水平 FOV 无效（应在 0-180 度之间）")
        
        # 验证平滑参数
        smoothing_factor = self.Get('smoothing.factor', 0)
        if smoothing_factor < 0 or smoothing_factor > 1:
            errors.append("平滑系数无效（应在 0-1 之间）")
        
        max_step = self.Get('smoothing.max_step', 0)
        if max_step <= 0:
            errors.append("最大步长无效（应大于 0）")
        
        # 验证 FPS
        fps = self.Get('detection.fps', 0)
        if fps <= 0 or fps > 120:
            errors.append("检测 FPS 无效（应在 1-120 之间）")
        
        return len(errors) == 0, errors

