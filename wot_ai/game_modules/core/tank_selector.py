#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克选择模块

从UI配置读取车辆优先级列表，使用模板匹配检测车辆可用性。
"""

import time
from pathlib import Path
from typing import Optional, List
from loguru import logger

from wot_ai.game_modules.ui_control.matcher_pyautogui import match_template
from wot_ai.config import get_program_dir


class TankSelector:
    """坦克选择器"""
    
    def __init__(self, vehicle_screenshot_dir: Path, vehicle_priority: List[str] = None):
        """
        初始化坦克选择器
        
        Args:
            vehicle_screenshot_dir: 车辆截图目录路径
            vehicle_priority: 车辆优先级列表（文件名列表），如果为None则按文件名排序
        """
        self.vehicle_screenshot_dir_ = Path(vehicle_screenshot_dir)
        self.vehicle_priority_ = vehicle_priority or []
        
        # 如果没有提供优先级列表，从目录中读取所有图片文件
        if not self.vehicle_priority_:
            if self.vehicle_screenshot_dir_.exists():
                image_files = sorted(self.vehicle_screenshot_dir_.glob("*.png"))
                image_files.extend(sorted(self.vehicle_screenshot_dir_.glob("*.jpg")))
                image_files.extend(sorted(self.vehicle_screenshot_dir_.glob("*.jpeg")))
                self.vehicle_priority_ = [f.name for f in image_files]
        
        logger.info(f"车辆优先级列表: {self.vehicle_priority_}")
    
    def pick(self, retry_interval: float = 10.0) -> Optional[Path]:
        """
        选择可用的车辆
        
        Args:
            retry_interval: 如果所有车辆都不可用，等待时间（秒）后重试
        
        Returns:
            可用车辆的截图路径，如果都不可用则返回None
        """
        while True:
            for vehicle_file in self.vehicle_priority_:
                vehicle_path = self.vehicle_screenshot_dir_ / vehicle_file
                
                if not vehicle_path.exists():
                    logger.warning(f"车辆截图不存在: {vehicle_path}")
                    continue
                
                # 使用模板匹配检测车辆是否在车库中可用
                center = match_template(vehicle_file, confidence=0.75, template_dir=str(get_program_dir() / "vehicle_screenshots"))
                
                if center is not None:
                    logger.info(f"找到可用车辆: {vehicle_file}")
                    return vehicle_path
                else:
                    logger.debug(f"车辆不可用（战斗中/冷却中）: {vehicle_file}")
            
            # 所有车辆都不可用，等待后重试
            logger.info(f"所有车辆都不可用，等待 {retry_interval} 秒后重试...")
            time.sleep(retry_interval)
    
    def update_priority(self, vehicle_priority: List[str]) -> None:
        """
        更新车辆优先级列表
        
        Args:
            vehicle_priority: 新的车辆优先级列表（文件名列表）
        """
        self.vehicle_priority_ = vehicle_priority
        logger.info(f"更新车辆优先级列表: {self.vehicle_priority_}")

