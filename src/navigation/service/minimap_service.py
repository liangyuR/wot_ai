#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图服务模块

封装小地图检测和初始化逻辑，管理小地图区域信息。
"""

from typing import Optional
import numpy as np
from loguru import logger

from src.vision.minimap_anchor_detector import MinimapAnchorDetector
from src.navigation.ui.transparent_overlay import TransparentOverlay
from src.navigation.config.models import NavigationConfig


class MinimapService:
    """小地图服务"""
    
    def __init__(self, anchor_detector: MinimapAnchorDetector, config: NavigationConfig):
        """
        初始化小地图服务
        
        Args:
            anchor_detector: 小地图锚点检测器
            config: NavigationConfig配置对象
        """
        self.anchor_detector_ = anchor_detector
        self.config_ = config
        self.minimap_region_: Optional[dict] = None
        self.overlay_: Optional[TransparentOverlay] = None
    
    def detect_region(self, frame: np.ndarray) -> Optional[dict]:
        """
        检测小地图区域
        
        Args:
            frame: 屏幕帧
        
        Returns:
            小地图区域字典 {x, y, width, height}，失败返回 None
        """
        # 获取屏幕尺寸
        frame_h, frame_w = frame.shape[:2]
        
        # 使用默认尺寸进行模板匹配（用于检测位置）
        default_size = self.config_.minimap.size
        
        # 使用MinimapAnchorDetector检测小地图位置
        top_left = self.anchor_detector_.detect(frame, size=default_size)
        if top_left is None:
            logger.warning("无法检测到小地图位置")
            return None
        
        x, y = top_left
        
        # 根据 top_left 到屏幕右下角的距离自适应计算小地图尺寸
        # 计算可用空间
        available_width = frame_w - x
        available_height = frame_h - y
        
        # 小地图通常是正方形，取宽度和高度的最小值
        minimap_size = min(available_width, available_height)
        
        # 确保尺寸合理（至少大于0，且不超过配置的最大值）
        minimap_size = max(1, min(minimap_size, self.config_.minimap.max_size))
        
        region = {
            'x': x,
            'y': y,
            'width': minimap_size,
            'height': minimap_size
        }
        
        logger.info(f"检测到小地图区域: {region} (自适应尺寸: {minimap_size}x{minimap_size})")
        self.minimap_region_ = region
        return region
    
    def initialize_overlay(self, minimap_region: dict) -> bool:
        """
        初始化透明覆盖层（直接绘制到小地图位置）
        
        Args:
            minimap_region: 小地图区域 {x, y, width, height}（用于确定 overlay 位置和尺寸）
        
        Returns:
            是否初始化成功
        """
        try:
            # overlay 直接绘制到小地图位置
            overlay_width = minimap_region['width']
            overlay_height = minimap_region['height']
            overlay_x = minimap_region['x']
            overlay_y = minimap_region['y']
            logger.info(f"初始化透明覆盖层，位置: ({overlay_x}, {overlay_y})（小地图位置）, "
                       f"尺寸: {overlay_width}x{overlay_height}")
            
            self.overlay_ = TransparentOverlay(
                width=overlay_width,
                height=overlay_height,
                window_name="WOT_AI Navigation Overlay",
                pos_x=overlay_x,  # 小地图X坐标
                pos_y=overlay_y,  # 小地图Y坐标
                fps=self.config_.ui.overlay_fps,
                alpha=self.config_.ui.overlay_alpha
            )
            
            logger.info("透明覆盖层初始化成功（位于小地图位置）")
            return True
            
        except Exception as e:
            logger.error(f"透明覆盖层初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @property
    def minimap_region(self) -> Optional[dict]:
        """
        获取小地图区域信息
        
        Returns:
            小地图区域字典，如果未检测到则返回None
        """
        return self.minimap_region_
    
    @property
    def overlay(self) -> Optional[TransparentOverlay]:
        """
        获取透明覆盖层实例
        
        Returns:
            TransparentOverlay实例，如果未初始化则返回None
        """
        return self.overlay_

