#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接绘制Overlay模块：使用Windows GDI直接绘制到屏幕DC

特性：
- 直接绘制到屏幕设备上下文，不会被mss捕获
- 完全透明，只绘制路径和标记
- 在小地图位置精确绘制
"""

from typing import Tuple, List, Optional
import threading
import time
import math
import hashlib
import numpy as np
import cv2

try:
    import win32gui
    import win32con
    import win32ui
    from win32api import RGB
except ImportError:
    win32gui = None
    win32con = None
    win32ui = None
    RGB = None

from loguru import logger


class DirectDrawOverlay:
    """直接绘制Overlay：使用Windows GDI直接绘制到屏幕DC"""
    
    def __init__(self, width: int, height: int,
                 pos_x: int = 0, pos_y: int = 0,
                 fps: int = 30):
        """
        初始化直接绘制Overlay
        
        Args:
            width: 绘制区域宽度
            height: 绘制区域高度
            pos_x: 绘制区域X坐标（屏幕坐标）
            pos_y: 绘制区域Y坐标（屏幕坐标）
            fps: 绘制帧率
        """
        if win32gui is None:
            raise ImportError("需要安装pywin32: pip install pywin32")
        
        self.width_ = width
        self.height_ = height
        self.pos_x_ = pos_x
        self.pos_y_ = pos_y
        self.fps_ = fps
        
        # 当前要显示的数据
        self.current_path_ = []
        self.current_path_hash_ = None
        self.cached_path_points_ = []  # 缓存的路径点（已转换的坐标）
        self.current_detections_ = {}
        self.current_mask_ = None
        self.minimap_size_ = (width, height)
        self.grid_size_ = (64, 64)
        
        # 绘制线程控制
        self.running_ = False
        self.draw_thread_ = None
        
        # 坐标转换缓存
        self._scale_cache_ = None
        self._grid_size_cache_ = None
        self._minimap_size_cache_ = None
        
        # 启动绘制线程
        self.Start_()
    
    def Start_(self):
        """启动绘制线程"""
        self.running_ = True
        self.draw_thread_ = threading.Thread(
            target=self._DrawThread_,
            daemon=True
        )
        self.draw_thread_.start()
        logger.info(f"直接绘制Overlay启动，位置: ({self.pos_x_}, {self.pos_y_}), 尺寸: {self.width_}x{self.height_}")
    
    def _DrawThread_(self):
        """绘制线程主循环"""
        target_dt = 1.0 / max(1, self.fps_)
        
        while self.running_:
            try:
                start_time = time.time()
                self._DrawFrame_()
                
                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, target_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"绘制线程错误: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("绘制线程退出")
    
    def _DrawFrame_(self):
        """绘制一帧"""
        if win32gui is None or win32ui is None:
            return
        
        # 获取屏幕DC
        hdc = win32gui.GetDC(0)
        if not hdc:
            return
        
        try:
            # 创建内存DC用于双缓冲
            mem_dc = win32ui.CreateCompatibleDC(hdc)
            bitmap = win32ui.CreateCompatibleBitmap(hdc, self.width_, self.height_)
            old_bitmap = mem_dc.SelectObject(bitmap)
            
            try:
                # 清空背景（透明，使用黑色）
                mem_dc.FillSolidRect((0, 0, self.width_, self.height_), RGB(0, 0, 0))
                
                # 绘制掩码（如果有）
                if self.current_mask_ is not None:
                    self._DrawMask_(mem_dc)
                
                # 绘制路径
                if self.cached_path_points_ and len(self.cached_path_points_) > 1:
                    self._DrawPath_(mem_dc)
                
                # 绘制起点（自己位置）
                if self.current_detections_.get('self_pos') is not None:
                    self._DrawSelfPosition_(mem_dc)
                
                # 绘制终点（基地位置）
                if self.current_detections_.get('flag_pos') is not None:
                    self._DrawFlagPosition_(mem_dc)
                
                # 将内存DC内容复制到屏幕DC（使用SRCPAINT模式实现透明效果）
                # 注意：这种方式可能仍会被mss捕获，但比窗口方式好
                win32gui.BitBlt(
                    hdc, self.pos_x_, self.pos_y_, self.width_, self.height_,
                    mem_dc.GetSafeHdc(), 0, 0, win32con.SRCCOPY
                )
                
            except Exception as e:
                logger.error(f"绘制元素时出错: {e}", exc_info=True)
            finally:
                mem_dc.SelectObject(old_bitmap)
                bitmap.DeleteObject()
                mem_dc.DeleteDC()
        
        finally:
            win32gui.ReleaseDC(0, hdc)
    
    def _DrawPath_(self, dc):
        """绘制路径"""
        if not self.cached_path_points_ or len(self.cached_path_points_) < 2:
            return
        
        try:
            # 创建绿色画笔
            pen = win32ui.CreatePen(win32con.PS_SOLID, 2, RGB(0, 255, 0))
            old_pen = dc.SelectObject(pen)
            
            try:
                # 绘制路径线段
                for i in range(len(self.cached_path_points_) - 1):
                    pt1 = self.cached_path_points_[i]
                    pt2 = self.cached_path_points_[i + 1]
                    dc.MoveTo((int(pt1[0]), int(pt1[1])))
                    dc.LineTo((int(pt2[0]), int(pt2[1])))
            finally:
                dc.SelectObject(old_pen)
                pen.DeleteObject()
        
        except Exception as e:
            logger.error(f"绘制路径时出错: {e}", exc_info=True)
    
    def _DrawSelfPosition_(self, dc):
        """绘制自己位置（起点）"""
        pos = self.current_detections_.get('self_pos')
        if pos is None:
            return
        
        try:
            x, y = int(pos[0]), int(pos[1])
            radius = 14
            
            # 创建黄色画笔和空画刷
            pen = win32ui.CreatePen(win32con.PS_SOLID, 3, RGB(255, 255, 0))
            brush = win32ui.CreateSolidBrush(RGB(0, 0, 0))  # 黑色填充（透明效果）
            old_pen = dc.SelectObject(pen)
            old_brush = dc.SelectObject(brush)
            
            try:
                # 绘制圆圈
                dc.Ellipse((x - radius, y - radius, x + radius, y + radius))
                
                # 绘制角度箭头（如果有角度信息）
                if self.current_detections_.get('self_angle') is not None:
                    angle_deg = self.current_detections_['self_angle']
                    angle_rad = math.radians(angle_deg)
                    
                    arrow_length = 30.0
                    end_x = x + arrow_length * math.cos(angle_rad)
                    end_y = y + arrow_length * math.sin(angle_rad)
                    
                    # 绘制箭头主线
                    dc.MoveTo((x, y))
                    dc.LineTo((int(end_x), int(end_y)))
                    
                    # 绘制箭头头部
                    head_length = 8.0
                    head_angle = math.pi / 6  # 30度
                    
                    head1_x = end_x - head_length * math.cos(angle_rad - head_angle)
                    head1_y = end_y - head_length * math.sin(angle_rad - head_angle)
                    head2_x = end_x - head_length * math.cos(angle_rad + head_angle)
                    head2_y = end_y - head_length * math.sin(angle_rad + head_angle)
                    
                    dc.MoveTo((int(end_x), int(end_y)))
                    dc.LineTo((int(head1_x), int(head1_y)))
                    dc.MoveTo((int(end_x), int(end_y)))
                    dc.LineTo((int(head2_x), int(head2_y)))
            
            finally:
                dc.SelectObject(old_pen)
                dc.SelectObject(old_brush)
                pen.DeleteObject()
                brush.DeleteObject()
        
        except Exception as e:
            logger.error(f"绘制自己位置时出错: {e}", exc_info=True)
    
    def _DrawFlagPosition_(self, dc):
        """绘制终点（基地位置）"""
        pos = self.current_detections_.get('flag_pos')
        if pos is None:
            return
        
        try:
            x, y = int(pos[0]), int(pos[1])
            radius = 8
            
            # 创建蓝色画笔和画刷
            pen = win32ui.CreatePen(win32con.PS_SOLID, 2, RGB(0, 0, 255))
            brush = win32ui.CreateSolidBrush(RGB(0, 0, 255))
            old_pen = dc.SelectObject(pen)
            old_brush = dc.SelectObject(brush)
            
            try:
                # 绘制外圈
                dc.Ellipse((x - radius, y - radius, x + radius, y + radius))
                
                # 绘制内圈（填充）
                inner_radius = 5
                dc.Ellipse((x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius))
            
            finally:
                dc.SelectObject(old_pen)
                dc.SelectObject(old_brush)
                pen.DeleteObject()
                brush.DeleteObject()
        
        except Exception as e:
            logger.error(f"绘制终点时出错: {e}", exc_info=True)
    
    def _DrawMask_(self, dc):
        """绘制掩码地图（障碍物用红色半透明显示）"""
        if self.current_mask_ is None:
            return
        
        try:
            minimap_w, minimap_h = self.minimap_size_
            
            # 将掩码resize到小地图尺寸
            mask_h, mask_w = self.current_mask_.shape[:2]
            mask_resized = cv2.resize(
                self.current_mask_.astype(np.uint8),
                (minimap_w, minimap_h),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 绘制障碍物区域（简化：只绘制主要障碍物块，降低分辨率以提高性能）
            obstacle_mask = (mask_resized == 1)
            step = 4  # 降低分辨率以提高性能
            for y in range(0, minimap_h, step):
                for x in range(0, minimap_w, step):
                    if obstacle_mask[y, x]:
                        dc.FillSolidRect((x, y, step, step), RGB(255, 0, 0))
        
        except Exception as e:
            logger.error(f"绘制掩码时出错: {e}", exc_info=True)
    
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
            grid_size: 栅格地图尺寸 (width, height)
            mask: 掩码地图（0=可通行，1=障碍）
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
                # 缓存缩放值
                self._scale_cache_ = (scale_x, scale_y)
                self._grid_size_cache_ = grid_size
                self._minimap_size_cache_ = minimap_size
            elif self._scale_cache_ and self._grid_size_cache_ and self._minimap_size_cache_:
                # 使用缓存的缩放值
                scale_x, scale_y = self._scale_cache_
            elif self.minimap_size_ and self.grid_size_:
                scale_x = self.minimap_size_[0] / float(self.grid_size_[0])
                scale_y = self.minimap_size_[1] / float(self.grid_size_[1])
            else:
                scale_x = scale_y = 1.0
                logger.warning("DrawPath: minimap_size和grid_size都未提供，使用默认缩放1.0")
            
            # 转换并缓存路径点
            if path and len(path) > 1:
                self.cached_path_points_ = [
                    (float(x * scale_x), float(y * scale_y))
                    for x, y in path
                ]
            else:
                self.cached_path_points_ = []
        
        # 实时更新检测结果
        if detections:
            for key, value in detections.items():
                if value is not None:
                    self.current_detections_[key] = value
        
        self.current_mask_ = mask
        if minimap_size:
            self.minimap_size_ = minimap_size
        elif minimap is not None:
            self.minimap_size_ = (minimap.shape[1], minimap.shape[0])
        if grid_size:
            self.grid_size_ = grid_size
    
    def SetPosition(self, x: int, y: int):
        """
        设置绘制位置
        
        Args:
            x: X坐标
            y: Y坐标
        """
        self.pos_x_ = x
        self.pos_y_ = y
    
    def SetFps(self, fps: int):
        """
        设置帧率
        
        Args:
            fps: 目标帧率
        """
        self.fps_ = fps
    
    def Hide(self):
        """隐藏绘制（停止绘制线程）"""
        self.running_ = False
        if self.draw_thread_ and self.draw_thread_.is_alive():
            self.draw_thread_.join(timeout=2.0)
    
    def Show(self):
        """显示绘制（启动绘制线程）"""
        if not self.running_:
            self.Start_()
    
    def IsVisible(self) -> bool:
        """
        检查是否可见
        
        Returns:
            是否可见
        """
        return self.running_
    
    def Close(self):
        """关闭绘制"""
        self.running_ = False
        if self.draw_thread_ and self.draw_thread_.is_alive():
            self.draw_thread_.join(timeout=2.0)
        logger.info("直接绘制Overlay已关闭")

