#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透明浮窗模块：使用 DearPyGui + Win32 在游戏画面上创建透明窗口显示路径
"""

# 标准库导入
from typing import Tuple, List, Optional
import threading
import hashlib

# 第三方库导入
import numpy as np
import cv2

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
        
        # 掩码背景纹理缓存
        self.mask_texture_id_ = None  # 掩码纹理ID
        self.mask_texture_hash_ = None  # 掩码哈希值，用于判断是否需要更新纹理
        
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
        elif self.current_path_ and len(self.current_path_) > 0:
            # 如果缓存为空但路径存在，可能是坐标转换问题，记录警告
            logger.warning(f"路径存在但缓存为空: 路径长度={len(self.current_path_)}, 缓存长度={len(self.cached_path_points_) if self.cached_path_points_ else 0}")
        
        # 实时绘制起点（自己位置）- 每帧更新
        if self.current_detections_.get('self_pos') is not None:
            pos = self.current_detections_['self_pos']
            x, y = float(pos[0]), float(pos[1])
            dpg.draw_circle((x, y), 14, color=(255, 255, 0, 255), thickness=3, fill=(0, 0, 0, 0), parent=drawlist_id)
            
            # 绘制当前角度箭头（如果角度可用）
            if self.current_detections_.get('self_angle') is not None:
                import math
                angle_deg = self.current_detections_['self_angle']
                angle_rad = math.radians(angle_deg)
                
                # 箭头长度（像素）
                arrow_length = 30.0
                
                # 计算箭头终点坐标
                # 注意：图像坐标系中y向下为正
                # extract_arrow_orientation返回的角度：0°向右，90°向下，180°向左，270°向上
                end_x = x + arrow_length * math.cos(angle_rad)
                end_y = y + arrow_length * math.sin(angle_rad)  # y向下为正，所以用加号
                
                # 绘制箭头主线
                dpg.draw_line(
                    (x, y),
                    (end_x, end_y),
                    color=(255, 255, 0, 255),  # 黄色
                    thickness=3.0,
                    parent=drawlist_id
                )
                
                # 绘制箭头头部（小三角形）
                # 箭头头部长度
                head_length = 8.0
                head_angle = math.pi / 6  # 30度
                
                # 计算箭头头部的两个点（从终点向起点方向偏移）
                head1_x = end_x - head_length * math.cos(angle_rad - head_angle)
                head1_y = end_y - head_length * math.sin(angle_rad - head_angle)
                head2_x = end_x - head_length * math.cos(angle_rad + head_angle)
                head2_y = end_y - head_length * math.sin(angle_rad + head_angle)
                
                # 绘制箭头头部（两条线）
                dpg.draw_line((end_x, end_y), (head1_x, head1_y), color=(255, 255, 0, 255), thickness=3.0, parent=drawlist_id)
                dpg.draw_line((end_x, end_y), (head2_x, head2_y), color=(255, 255, 0, 255), thickness=3.0, parent=drawlist_id)
        
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
    
    def _UpdateMaskTexture_(self, mask: np.ndarray):
        """
        更新掩码背景纹理
        
        Args:
            mask: 掩码地图（0=可通行，1=障碍）
        """
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            return
        
        if mask is None or mask.size == 0:
            return
        
        minimap_w, minimap_h = self.minimap_size_
        
        # 计算掩码哈希值，判断是否需要更新纹理
        mask_hash = hashlib.md5(mask.tobytes()).hexdigest()
        
        # 如果掩码未变化，不需要更新
        if mask_hash == self.mask_texture_hash_ and self.mask_texture_id_ is not None:
            return
        
        # 将掩码resize到小地图尺寸
        mask_h, mask_w = mask.shape[:2]
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (minimap_w, minimap_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # 创建RGBA图像：障碍物区域为红色半透明，其他区域透明
        mask_rgba = np.zeros((minimap_h, minimap_w, 4), dtype=np.uint8)
        obstacle_mask = (mask_resized == 1)
        alpha_value = int(255 * self.mask_alpha_)
        mask_rgba[obstacle_mask] = [255, 0, 0, alpha_value]  # 红色，半透明
        
        # 转换为DPG需要的格式（RGBA，flatten，归一化到0-1）
        mask_flat = mask_rgba.flatten().astype(np.float32) / 255.0
        
        # 删除旧纹理（如果存在）
        if self.mask_texture_id_ is not None:
            try:
                dpg.delete_item(self.mask_texture_id_)
            except Exception:
                pass
        
        # 创建新纹理（需要在texture_registry上下文中）
        texture_tag = f"mask_bg_texture_{id(self)}"
        with dpg.texture_registry():
            self.mask_texture_id_ = dpg.add_raw_texture(
                width=minimap_w,
                height=minimap_h,
                default_value=mask_flat,
                format=dpg.mvFormat_Float_rgba,
                tag=texture_tag
            )
        
        self.mask_texture_hash_ = mask_hash
    
    def _DrawMask_(self, drawlist_id: int):
        """
        绘制掩码地图作为背景（障碍物用红色半透明显示）
        
        使用纹理贴图方式，将掩码resize后直接绘制，性能更好。
        
        Args:
            drawlist_id: DearPyGui drawlist ID
        """
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            return
        
        # 绘制背景纹理（如果存在）
        if self.mask_texture_id_ is not None:
            minimap_w, minimap_h = self.minimap_size_
            dpg.draw_image(
                self.mask_texture_id_,
                pmin=(0, 0),
                pmax=(minimap_w, minimap_h),
                parent=drawlist_id
            )

    
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
                # 如果都没有，使用默认缩放（可能坐标已经是像素坐标）
                scale_x = scale_y = 1.0
                logger.warning("DrawPath: minimap_size和grid_size都未提供，使用默认缩放1.0")
            
            # 转换并缓存路径点
            if path and len(path) > 1:
                self.cached_path_points_ = [
                    (float(x * scale_x), float(y * scale_y)) 
                    for x, y in path
                ]
                logger.debug(f"路径点已缓存: {len(self.cached_path_points_)} 个点, scale=({scale_x:.2f}, {scale_y:.2f})")
            else:
                self.cached_path_points_ = []
                logger.debug("路径为空，清空缓存路径点")
        
        # 实时更新检测结果（self_pose等）
        self.current_detections_ = detections
        self.current_mask_ = mask
        if minimap_size:
            self.minimap_size_ = minimap_size
        elif minimap is not None:
            self.minimap_size_ = (minimap.shape[1], minimap.shape[0])
        if grid_size:
            self.grid_size_ = grid_size
        
        # 如果掩码更新，更新背景纹理
        if mask is not None:
            self._UpdateMaskTexture_(mask)
    
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

