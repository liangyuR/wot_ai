#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克选择模块

从UI配置读取车辆优先级列表，使用模板匹配检测车辆可用性。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from loguru import logger

@dataclass(frozen=True)
class TankTemplate:
    """坦克模板信息"""
    
    name: str
    path: Path
    confidence: float = 0.75
    
    @property
    def directory(self) -> Path:
        """模板所在目录"""
        return self.path.parent


class TankSelector:
    """坦克选择器"""
    
    _instance: Optional['TankSelector'] = None
    
    def __new__(cls) -> 'TankSelector':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化坦克选择器
        
        Args:
            vehicle_screenshot_dir: 车辆截图目录路径
            vehicle_priority: 车辆优先级列表（文件名列表），如果为None则按文件名排序
        """
        # 如果已经初始化过，跳过
        if hasattr(self, '_initialized'):
            return
        
        from src.utils.global_path import GetVehicleScreenshotsDir
        self.vehicle_screenshot_dir_ = GetVehicleScreenshotsDir()
        self.vehicle_priority_ = []
        
        # 从目录中读取所有图片文件
        if not self.vehicle_priority_ and self.vehicle_screenshot_dir_.exists():
            image_files = sorted(self.vehicle_screenshot_dir_.glob("*.png"))
            self.vehicle_priority_ = [f.name for f in image_files]
        
        logger.info(f"车辆优先级列表: {self.vehicle_priority_}")
        self._initialized = True
    
    def pick(self) -> List[TankTemplate]:
        """
        返回按优先级排序的车辆模板信息列表
        
        Returns:
            TankTemplate 列表
        """
        templates: List[TankTemplate] = []
        for vehicle_file in self.vehicle_priority_:
            vehicle_path = self.vehicle_screenshot_dir_ / vehicle_file
            if not vehicle_path.exists():
                logger.warning(f"车辆截图不存在: {vehicle_path}")
                continue
            
            templates.append(TankTemplate(name=vehicle_file, path=vehicle_path))
        
        if not templates:
            logger.error("车辆模板列表为空，请检查车辆截图目录")
        
        return templates
    
    def update_priority(self, vehicle_priority: List[str]) -> None:
        """
        更新车辆优先级列表
        
        Args:
            vehicle_priority: 新的车辆优先级列表（文件名列表）
        """
        self.vehicle_priority_ = vehicle_priority
        logger.info(f"更新车辆优先级列表: {self.vehicle_priority_}")

