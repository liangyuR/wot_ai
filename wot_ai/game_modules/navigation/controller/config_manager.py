#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器：负责配置文件的加载、验证、合并
"""

# 标准库导入
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml

# 本地模块导入
from ..common.imports import GetLogger
from ..common.config import NavigationConfig

logger = GetLogger()(__name__)


class ConfigManager:
    """配置管理器：处理配置文件的加载、验证和合并"""
    
    def __init__(self, config_path: Path):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path_ = config_path
        self.config_: Dict[str, Any] = {}
    
    def LoadConfig(self) -> bool:
        """
        加载配置文件
        
        Returns:
            是否加载成功
        """
        try:
            if not self.config_path_.exists():
                logger.warning(f"配置文件不存在: {self.config_path_}，使用默认配置")
                self.config_ = self.GetDefaultConfig()
                return True
            
            with open(self.config_path_, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f) or {}
            
            # 合并默认配置
            default_config = self.GetDefaultConfig()
            self.config_ = self.MergeConfig(default_config, loaded_config)
            
            # 验证配置
            is_valid, error_msg = self.ValidateConfig(self.config_)
            if not is_valid:
                logger.warning(f"配置验证失败: {error_msg}，继续使用配置")
            
            logger.info(f"配置加载成功: {self.config_path_}")
            return True
        except Exception as e:
            logger.error(f"加载配置失败: {e}，使用默认配置")
            self.config_ = self.GetDefaultConfig()
            return False
    
    def GetDefaultConfig(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return NavigationConfig.GetDefaultConfig()
    
    def ValidateConfig(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证配置的有效性
        
        Args:
            config: 配置字典
        
        Returns:
            (是否有效, 错误信息)
        """
        return NavigationConfig.ValidateConfig(config)
    
    def MergeConfig(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并配置字典
        
        Args:
            default: 默认配置
            override: 覆盖配置
        
        Returns:
            合并后的配置
        """
        return NavigationConfig.MergeConfig(default, override)
    
    def GetConfig(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        Returns:
            配置字典
        """
        return self.config_

