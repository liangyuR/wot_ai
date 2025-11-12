#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透明浮窗模块：使用 DearPyGui + Win32 在游戏画面上创建透明窗口显示路径
"""

# 标准库导入
from typing import Tuple, List
import threading

# 第三方库导入
import numpy as np

# 本地模块导入
from ..common.imports import GetLogger
from ..common.overlay_config import OverlayConfig

logger = GetLogger()(__name__)

# 导入 DPG Overlay 管理器
from .overlay_dpg import DPGOverlayManager


class TransparentOverlay:
    """透明浮窗：在游戏画面上叠加显示路径（基于 DearPyGui）"""
    
    def __init__(self, width: int, height: int, 
                 window_name: str = "WOT_AI Overlay",
                 pos_x: int = 0, pos_y: int = 0,
                 fps: int = 30, alpha: int = 180):
        """
        初始化透明浮窗
        
        Args:
            width: 窗口宽度
            height: 窗口高度
            window_name: 窗口名称
            pos_x: 窗口X坐标
            pos_y: 窗口Y坐标
            fps: 帧率
            alpha: 透明度 (0-255)
        """
        self.width_ = width
        self.height_ = height
        self.window_name_ = window_name
        self.pos_x_ = pos_x
        self.pos_y_ = pos_y
        self.fps_ = fps
        self.alpha_ = alpha
        
        # 当前要显示的数据
        self.current_path_ = []
        self.current_detections_ = {}
        self.minimap_size_ = (width, height)
        self.grid_size_ = (64, 64)  # 默认栅格大小
        
        # DPG Overlay 管理器
        self.overlay_manager_ = None
        self.overlay_thread_ = None
        self.running_ = False
        
        # 启动 Overlay
        self.StartOverlay_()
    
    def StartOverlay_(self):
        """在后台线程中启动 Overlay"""
        config = OverlayConfig(
            width=self.width_,
            height=self.height_,
            pos_x=self.pos_x_,
            pos_y=self.pos_y_,
            fps=self.fps_,
            alpha=self.alpha_,
            title=self.window_name_,
            target_window_title=None,  # 可以后续绑定到游戏窗口
            click_through=True
        )
        
        self.overlay_manager_ = DPGOverlayManager(config)
        
        # 在后台线程中运行
        self.overlay_thread_ = threading.Thread(
            target=self.OverlayThread_,
            daemon=True
        )
        self.overlay_thread_.start()
    
    def OverlayThread_(self):
        """Overlay 线程主循环"""
        self.running_ = True
        self.overlay_manager_.Run(self.DrawCallback_)
        self.running_ = False
    
    def DrawCallback_(self, drawlist_id: int, mgr: DPGOverlayManager):
        """
        DearPyGui 绘制回调
        
        Args:
            drawlist_id: DearPyGui drawlist ID
            mgr: Overlay 管理器实例
        """
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            return
        
        # 绘制路径
        if self.current_path_ and len(self.current_path_) > 1:
            # 计算坐标缩放（从栅格坐标到像素坐标）
            if self.minimap_size_ and self.grid_size_:
                scale_x = self.minimap_size_[0] / float(self.grid_size_[0])
                scale_y = self.minimap_size_[1] / float(self.grid_size_[1])
                coord_scale = (scale_x, scale_y)
            else:
                coord_scale = None
            
            mgr.DrawPath(
                drawlist_id,
                self.current_path_,
                color=(0, 255, 0, 255),  # 绿色路径
                thickness=2.0,
                coord_scale=coord_scale
            )
        
        # 绘制起点（自己位置）
        if self.current_detections_.get('self_pos') is not None:
            pos = self.current_detections_['self_pos']
            x, y = float(pos[0]), float(pos[1])
            dpg.draw_circle((x, y), 8, color=(255, 255, 0, 255), thickness=2, parent=drawlist_id)
            dpg.draw_circle((x, y), 5, color=(255, 255, 0, 255), fill=(255, 255, 0, 255), parent=drawlist_id)
        
        # 绘制终点（基地位置）
        if self.current_detections_.get('flag_pos') is not None:
            pos = self.current_detections_['flag_pos']
            x, y = float(pos[0]), float(pos[1])
            dpg.draw_circle((x, y), 8, color=(0, 0, 255, 255), thickness=2, parent=drawlist_id)
            dpg.draw_circle((x, y), 5, color=(0, 0, 255, 255), fill=(0, 0, 255, 255), parent=drawlist_id)
    
    def Draw(self, img: np.ndarray):
        """
        在透明窗口上绘制图像（兼容接口，实际使用 DrawPath）
        
        Args:
            img: BGR 格式的 numpy 数组（此方法保留兼容性，实际绘制通过 DrawPath）
        """
        # DearPyGui 方案不需要这个，但保留接口兼容
        pass
    
    def DrawPath(self, minimap: np.ndarray, path: List[Tuple[int, int]], 
                 detections: dict, minimap_size: Tuple[int, int] = None,
                 grid_size: Tuple[int, int] = None):
        """
        在小地图上绘制路径（主要接口）
        
        Args:
            minimap: 小地图图像（BGR格式，用于参考）
            path: 路径坐标列表 [(x, y), ...]（栅格坐标）
            detections: 检测结果字典
            minimap_size: 小地图尺寸 (width, height)
            grid_size: 栅格地图尺寸 (width, height)，默认 (64, 64)
        """
        self.current_path_ = path
        self.current_detections_ = detections
        if minimap_size:
            self.minimap_size_ = minimap_size
        elif minimap is not None:
            self.minimap_size_ = (minimap.shape[1], minimap.shape[0])
        if grid_size:
            self.grid_size_ = grid_size
    
    def SetPosition(self, x: int, y: int):
        """
        设置窗口位置
        
        Args:
            x: X坐标
            y: Y坐标
        """
        self.pos_x_ = x
        self.pos_y_ = y
        
        # DearPyGui 的窗口位置在初始化时设置，这里可以绑定到游戏窗口
        if self.overlay_manager_:
            # 可以通过 attach_to_window 来绑定
            pass
    
    def SetFps(self, fps: int):
        """
        设置帧率
        
        Args:
            fps: 目标帧率
        """
        self.fps_ = fps
        if self.overlay_manager_:
            self.overlay_manager_.config_.fps = fps
    
    def Close(self):
        """关闭窗口"""
        self.running_ = False
        if self.overlay_manager_:
            try:
                import dearpygui.dearpygui as dpg
                # 停止 DPG 主循环
                dpg.stop_dearpygui()
            except Exception as e:
                logger.error(f"关闭窗口失败: {e}")
        
        if self.overlay_thread_ and self.overlay_thread_.is_alive():
            self.overlay_thread_.join(timeout=2.0)
        
        logger.info("透明浮窗已关闭")

