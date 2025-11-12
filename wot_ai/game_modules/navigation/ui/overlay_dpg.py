#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DearPyGui + Win32 透明 Overlay

特性：
- 置顶无边框窗口
- 支持全局透明度
- 鼠标穿透（不影响游戏操作）
- 每帧回调绘制（画路径、点、文字等）
"""

# 标准库导入
from typing import Callable, Optional, Tuple
import time

# 统一导入机制
from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple
setup_python_path()

SetupLogger = None
logger_module, _ = try_import_multiple([
    'wot_ai.game_modules.common.utils.logger',
    'game_modules.common.utils.logger',
    'common.utils.logger',
    'yolo.utils.logger'
])
if logger_module is not None:
    SetupLogger = getattr(logger_module, 'SetupLogger', None)

if SetupLogger is None:
    from ...common.utils.logger import SetupLogger

logger = SetupLogger(__name__)

# 本地模块导入
from ..common.overlay_config import OverlayConfig


class DPGOverlayManager:
    """
    DearPyGui 透明 Overlay 管理器
    
    用法：
        mgr = DPGOverlayManager(config)
        mgr.run(draw_callback)
    
    draw_callback 会在每一帧被调用：
        def draw_callback(drawlist_id, mgr):
            # 在 drawlist_id 上画任意东西
    """
    
    def __init__(self, config: OverlayConfig):
        self.config_ = config
        self.hwnd_ = None
        self.drawlist_id_ = None
        self.running_ = False
    
    def _InitDpg(self):
        """初始化 DPG + 窗口"""
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            logger.error("dearpygui 未安装，请运行: pip install dearpygui")
            raise
        
        try:
            import win32gui
            import win32con
        except ImportError:
            logger.error("pywin32 未安装，请运行: pip install pywin32")
            raise
        
        dpg.create_context()
        
        # 创建无边框、置顶 viewport
        dpg.create_viewport(
            title=self.config_.title,
            width=self.config_.width,
            height=self.config_.height,
            x_pos=self.config_.pos_x,
            y_pos=self.config_.pos_y,
            decorated=False,
            always_on_top=True
        )
        dpg.setup_dearpygui()
        
        # 主窗口（占满整个 viewport）
        with dpg.window(
            tag="overlay_window",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_background=True,   # 让背景透明，主要看全局 alpha
        ):
            self.drawlist_id_ = dpg.add_drawlist(
                width=self.config_.width,
                height=self.config_.height,
                tag="overlay_drawlist"
            )
        
        dpg.set_primary_window("overlay_window", True)
        dpg.show_viewport()
        
        # 获取底层窗口句柄（DearPyGui 提供）
        try:
            self.hwnd_ = dpg.get_viewport_platform_handle()
        except Exception:
            # 极端情况下用 FindWindow 兜底
            self.hwnd_ = win32gui.FindWindow(None, self.config_.title)
        
        if not self.hwnd_:
            logger.error("无法获取 Overlay 窗口句柄")
            return
        
        # 设置窗口位置（确保定位到小地图位置）
        self._SetWindowPosition()
        
        # 设置透明 + 鼠标穿透
        self._ApplyWin32Style()
        
        # 如果配置了目标窗口标题，则绑定到游戏窗口位置
        if self.config_.target_window_title:
            self.AttachToWindow(self.config_.target_window_title)
    
    def _SetWindowPosition(self):
        """设置窗口位置到指定坐标"""
        if not self.hwnd_:
            return
        
        try:
            import win32gui
            import win32con
        except ImportError:
            return
        
        # 使用 MoveWindow 设置窗口位置和大小
        win32gui.SetWindowPos(
            self.hwnd_,
            win32con.HWND_TOPMOST,
            self.config_.pos_x,
            self.config_.pos_y,
            self.config_.width,
            self.config_.height,
            win32con.SWP_SHOWWINDOW
        )
        
        logger.info(f"Overlay 窗口定位到: ({self.config_.pos_x}, {self.config_.pos_y}) "
                   f"尺寸: {self.config_.width}x{self.config_.height}")
    
    def _ApplyWin32Style(self):
        """设置 Win32 样式：透明 + 穿透"""
        if not self.hwnd_:
            return
        
        try:
            import win32gui
            import win32con
        except ImportError:
            return
        
        ex_style = win32gui.GetWindowLong(self.hwnd_, win32con.GWL_EXSTYLE)
        
        # 分层窗口 + 置顶
        ex_style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST
        
        # 鼠标穿透（重要）
        if self.config_.click_through:
            ex_style |= win32con.WS_EX_TRANSPARENT
        
        win32gui.SetWindowLong(self.hwnd_, win32con.GWL_EXSTYLE, ex_style)
        
        # 设置全局 alpha：0(完全透明)~255(不透明)
        alpha = max(0, min(255, self.config_.alpha))
        win32gui.SetLayeredWindowAttributes(
            self.hwnd_,
            0,                      # 颜色键，这里不用
            alpha,
            win32con.LWA_ALPHA      # 用全局 Alpha
        )
    
    def AttachToWindow(self, title: str):
        """让 Overlay 贴在某个窗口上，比如 World of Tanks"""
        try:
            import win32gui
            import dearpygui.dearpygui as dpg
        except ImportError:
            return
        
        hwnd_game = win32gui.FindWindow(None, title)
        if not hwnd_game:
            logger.warning(f"未找到目标窗口: {title!r}")
            return
        
        left, top, right, bottom = win32gui.GetWindowRect(hwnd_game)
        width = right - left
        height = bottom - top
        
        logger.info(f"绑定 Overlay 到窗口 {title!r} rect={left, top, right, bottom}")
        
        # 调整 DearPyGui viewport
        dpg.set_viewport_pos((left, top))
        dpg.set_viewport_width(width)
        dpg.set_viewport_height(height)
        
        # 调整 Win32 窗口大小/位置
        if self.hwnd_:
            win32gui.MoveWindow(self.hwnd_, left, top, width, height, True)
    
    def Run(self, draw_callback: Optional[Callable[[int, "DPGOverlayManager"], None]] = None):
        """
        启动 Overlay 主循环
        
        draw_callback(drawlist_id, self) 会在每一帧调用，
        你可以在里面画路径、点、文字等。
        """
        if self.running_:
            logger.warning("Overlay 已经在运行")
            return
        
        self._InitDpg()
        
        if not self.hwnd_:
            logger.error("Overlay 初始化失败，退出")
            return
        
        self.running_ = True
        logger.info("DPG Overlay 启动")
        
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            return
        
        target_dt = 1.0 / max(1, self.config_.fps)
        
        try:
            while dpg.is_dearpygui_running():
                start = time.time()
                
                # 每帧先清空 drawlist
                if self.drawlist_id_ is not None:
                    dpg.delete_item(self.drawlist_id_, children_only=True)
                
                # 让上层逻辑绘制
                if draw_callback is not None and self.drawlist_id_ is not None:
                    try:
                        draw_callback(self.drawlist_id_, self)
                    except Exception as e:
                        logger.error(f"Overlay 绘制回调异常: {e}", exc_info=True)
                
                dpg.render_dearpygui_frame()
                
                # 简单帧率控制
                elapsed = time.time() - start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        finally:
            self.running_ = False
            logger.info("DPG Overlay 结束")
            dpg.destroy_context()
    
    def DrawPath(self, drawlist_id: int, path, color=(0, 255, 0, 255), 
                 thickness: float = 2.0, coord_scale: Optional[Tuple[float, float]] = None):
        """
        帮你把 A* 栅格路径画到屏幕坐标上。
        
        Args:
            drawlist_id: DearPyGui drawlist ID
            path: [(x, y), ...] 栅格坐标或小地图坐标
            color: RGBA 颜色元组
            thickness: 线条粗细
            coord_scale: (sx, sy)，如果你想从 [0,1] 或小地图尺寸映射到屏幕
        """
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            return
        
        if not path or len(path) < 2:
            return
        
        points = []
        if coord_scale is None:
            # 默认当成已经是屏幕坐标
            points = [(float(x), float(y)) for x, y in path]
        else:
            sx, sy = coord_scale
            for x, y in path:
                points.append((float(x * sx), float(y * sy)))
        
        # 连线
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            dpg.draw_line((x1, y1), (x2, y2), color=color, thickness=thickness, parent=drawlist_id)
        
        # 起点终点画一下圈
        sx, sy = points[0]
        ex, ey = points[-1]
        dpg.draw_circle((sx, sy), 6, color=(0, 255, 0, 255), thickness=2, parent=drawlist_id)
        dpg.draw_circle((ex, ey), 6, color=(255, 0, 0, 255), thickness=2, parent=drawlist_id)

