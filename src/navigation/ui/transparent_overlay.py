#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透明浮窗模块：使用Windows GDI直接绘制到屏幕DC显示路径

特性：
- 直接绘制到屏幕设备上下文，不会被mss捕获
- 完全透明，只绘制路径和标记
- 在小地图位置精确绘制
"""

# 标准库导入
from typing import Tuple, List, Optional

# 第三方库导入
import numpy as np

# 本地模块导入
from loguru import logger

# 导入直接绘制Overlay
from .direct_draw_overlay import DirectDrawOverlay


class TransparentOverlay:
    """透明浮窗：在游戏画面上叠加显示路径（基于Windows GDI直接绘制）"""
    
    def __init__(self, width: int, height: int, 
                 window_name: str = "WOT_AI Overlay",
                 pos_x: int = 0, pos_y: int = 0,
                 fps: int = 30, alpha: int = 180):
        """
        初始化透明浮窗
        
        Args:
            width: 窗口宽度
            height: 窗口高度
            window_name: 窗口名称（保留兼容性，实际不使用）
            pos_x: 窗口X坐标
            pos_y: 窗口Y坐标
            fps: 帧率
            alpha: 透明度 (0-255)（保留兼容性，实际不使用）
        """
        self.width_ = width
        self.height_ = height
        self.window_name_ = window_name
        self.pos_x_ = pos_x
        self.pos_y_ = pos_y
        self.fps_ = fps
        self.alpha_ = alpha
        
        # 直接绘制Overlay实例
        self.direct_overlay_ = DirectDrawOverlay(
            width=width,
            height=height,
            pos_x=pos_x,
            pos_y=pos_y,
            fps=fps
        )

    
    def DrawPath(self, minimap: Optional[np.ndarray], path: List[Tuple[int, int]], 
                 detections: dict, minimap_size: Tuple[int, int] = None,
                 grid_size: Tuple[int, int] = None, mask: np.ndarray = None):
        """
        在小地图上绘制路径（主要接口）
        
        Args:
            minimap: 小地图图像（可选，主要用于获取尺寸）
            path: 路径坐标列表 [(x, y), ...]（栅格坐标）
            detections: 检测结果字典
            minimap_size: 小地图尺寸 (width, height)
            grid_size: 栅格地图尺寸 (width, height)，默认 (64, 64)
            mask: 掩码地图（0=可通行，1=障碍），尺寸为 (H, W)
        """
        # 委托给直接绘制Overlay
        self.direct_overlay_.DrawPath(
            minimap=minimap,
            path=path,
            detections=detections,
            minimap_size=minimap_size,
            grid_size=grid_size,
            mask=mask
        )
    
    def SetPosition(self, x: int, y: int):
        """
        设置窗口位置
        
        Args:
            x: X坐标
            y: Y坐标
        """
        self.pos_x_ = x
        self.pos_y_ = y
        if self.direct_overlay_:
            self.direct_overlay_.SetPosition(x, y)
    
    def SetFps(self, fps: int):
        """
        设置帧率
        
        Args:
            fps: 目标帧率
        """
        self.fps_ = fps
        if self.direct_overlay_:
            self.direct_overlay_.SetFps(fps)
    
    def Hide(self):
        """隐藏窗口"""
        if self.direct_overlay_:
            self.direct_overlay_.Hide()
    
    def Show(self):
        """显示窗口"""
        if self.direct_overlay_:
            self.direct_overlay_.Show()
    
    def IsVisible(self) -> bool:
        """
        检查窗口是否可见
        
        Returns:
            窗口是否可见
        """
        if self.direct_overlay_:
            return self.direct_overlay_.IsVisible()
        return False
    
    def Close(self):
        """关闭窗口"""
        if self.direct_overlay_:
            self.direct_overlay_.Close()
        logger.info("透明浮窗已关闭")

