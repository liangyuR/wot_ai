#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透明浮窗模块：使用 DearPyGui + Win32 在游戏画面上创建透明窗口显示路径
"""

# 标准库导入
from typing import Tuple, List, Optional
import threading

# 第三方库导入
import numpy as np

# 本地模块导入
from ..common.overlay_config import OverlayConfig
from loguru import logger

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
        self.current_path_hash_ = None  # 路径哈希值，用于判断路径是否变化
        self.cached_path_points_ = []  # 缓存的路径点（已转换的坐标）
        self.current_detections_ = {}
        self.current_mask_ = None  # 掩码地图（0=可通行，1=障碍）
        self.minimap_size_ = (width, height)
        self.grid_size_ = (64, 64)  # 默认栅格大小
        self.mask_alpha_ = 0.3  # 掩码透明度（0.0-1.0）
        
        # FPS显示
        self.current_ui_fps_ = 0.0  # UI线程FPS
        self.current_detection_fps_ = 0.0  # 检测线程FPS
        
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
        
        # 绘制掩码地图（如果有，在路径之前绘制）
        if self.current_mask_ is not None:
            self._DrawMask_(drawlist_id)
        
        # 绘制路径（使用缓存的路径点，只在路径变化时重新计算）
        if self.cached_path_points_ and len(self.cached_path_points_) > 1:
            # 使用缓存的路径点直接绘制
            for i in range(len(self.cached_path_points_) - 1):
                pt1 = self.cached_path_points_[i]
                pt2 = self.cached_path_points_[i + 1]
                dpg.draw_line(pt1, pt2, color=(0, 255, 0, 255), thickness=2.0, parent=drawlist_id)
        
        # 实时绘制起点（自己位置）- 每帧更新
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
        
        # 绘制FPS信息（左上角）
        fps_text = f"UI: {self.current_ui_fps_:.1f} FPS"
        if self.current_detection_fps_ > 0:
            fps_text += f" | Detect: {self.current_detection_fps_:.1f} FPS"
        dpg.draw_text((10, 10), fps_text, color=(255, 255, 255, 255), size=16, parent=drawlist_id)
    
    def _DrawMask_(self, drawlist_id: int):
        """
        绘制掩码地图（障碍物用红色半透明显示）
        
        Args:
            drawlist_id: DearPyGui drawlist ID
        """
        # 暂时禁用掩码绘制以提升性能
        return
        
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            return
        
        if self.current_mask_ is None or self.current_mask_.size == 0:
            return
        
        mask_h, mask_w = self.current_mask_.shape[:2]
        minimap_w, minimap_h = self.minimap_size_
        
        # 计算缩放比例
        scale_x = minimap_w / float(mask_w)
        scale_y = minimap_h / float(mask_h)
        
        # 障碍物颜色（红色，半透明）
        obstacle_color = (255, 0, 0, int(255 * self.mask_alpha_))
        
        # 遍历掩码，绘制障碍物区域
        # 为了性能，可以采样绘制（每N个像素绘制一次）
        step = max(1, int(min(scale_x, scale_y)))  # 根据缩放比例调整采样步长
        
        for y in range(0, mask_h, step):
            for x in range(0, mask_w, step):
                if self.current_mask_[y, x] == 1:  # 障碍物
                    # 转换为小地图坐标
                    px = x * scale_x
                    py = y * scale_y
                    # 绘制小矩形表示障碍物
                    size = max(2, int(min(scale_x, scale_y)))
                    dpg.draw_rectangle(
                        (px, py),
                        (px + size, py + size),
                        color=obstacle_color,
                        fill=obstacle_color,
                        parent=drawlist_id
                    )
    
    def Draw(self, img: np.ndarray):
        """
        在透明窗口上绘制图像（兼容接口，实际使用 DrawPath）
        
        Args:
            img: BGR 格式的 numpy 数组（此方法保留兼容性，实际绘制通过 DrawPath）
        """
        # DearPyGui 方案不需要这个，但保留接口兼容
        pass
    
    def DrawPath(self, minimap: Optional[np.ndarray], path: List[Tuple[int, int]], 
                 detections: dict, minimap_size: Tuple[int, int] = None,
                 grid_size: Tuple[int, int] = None, mask: np.ndarray = None):
        """
        在小地图上绘制路径（主要接口）
        
        Args:
            minimap: 小地图图像（BGR格式，可选，主要用于获取尺寸，如果提供了minimap_size则不需要）
            path: 路径坐标列表 [(x, y), ...]（栅格坐标）
            detections: 检测结果字典
            minimap_size: 小地图尺寸 (width, height)
            grid_size: 栅格地图尺寸 (width, height)，默认 (64, 64)
            mask: 掩码地图（0=可通行，1=障碍），尺寸为 (H, W)
        """
        # 计算路径哈希值，判断路径是否变化
        import hashlib
        path_str = str(path) if path else ""
        path_hash = hashlib.md5(path_str.encode()).hexdigest()
        
        # 如果路径变化，重新计算并缓存路径点
        if path_hash != self.current_path_hash_:
            self.current_path_ = path
            self.current_path_hash_ = path_hash
            
            # 计算坐标缩放（从栅格坐标到像素坐标）
            if minimap_size and grid_size:
                scale_x = minimap_size[0] / float(grid_size[0])
                scale_y = minimap_size[1] / float(grid_size[1])
            elif self.minimap_size_ and self.grid_size_:
                scale_x = self.minimap_size_[0] / float(self.grid_size_[0])
                scale_y = self.minimap_size_[1] / float(self.grid_size_[1])
            else:
                scale_x = scale_y = 1.0
            
            # 转换并缓存路径点
            if path and len(path) > 1:
                self.cached_path_points_ = [
                    (float(x * scale_x), float(y * scale_y)) 
                    for x, y in path
                ]
            else:
                self.cached_path_points_ = []
        
        # 实时更新检测结果（self_pose等）
        self.current_detections_ = detections
        self.current_mask_ = mask
        if minimap_size:
            self.minimap_size_ = minimap_size
        elif minimap is not None:
            self.minimap_size_ = (minimap.shape[1], minimap.shape[0])
        if grid_size:
            self.grid_size_ = grid_size
    
    def UpdateFps(self, ui_fps: float, detection_fps: float = 0.0):
        """
        更新FPS显示
        
        Args:
            ui_fps: UI线程FPS
            detection_fps: 检测线程FPS（可选）
        """
        self.current_ui_fps_ = ui_fps
        self.current_detection_fps_ = detection_fps
    
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
    
    def Hide(self):
        """隐藏窗口"""
        if self.overlay_manager_:
            self.overlay_manager_.HideWindow()
    
    def Show(self):
        """显示窗口"""
        if self.overlay_manager_:
            self.overlay_manager_.ShowWindow()
    
    def IsVisible(self) -> bool:
        """
        检查窗口是否可见
        
        Returns:
            窗口是否可见
        """
        if self.overlay_manager_:
            return self.overlay_manager_.IsVisible()
        return False
    
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

