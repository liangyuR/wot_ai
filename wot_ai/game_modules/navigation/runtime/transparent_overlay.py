#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透明浮窗模块：在游戏画面上创建透明窗口显示路径
"""

from typing import Optional, Tuple, List
import numpy as np
import cv2

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


class TransparentOverlay:
    """透明浮窗：在游戏画面上叠加显示路径"""
    
    def __init__(self, width: int, height: int, 
                 window_name: str = "WOT_AI Overlay",
                 pos_x: int = 0, pos_y: int = 0):
        """
        初始化透明浮窗
        
        Args:
            width: 窗口宽度
            height: 窗口高度
            window_name: 窗口名称
            pos_x: 窗口X坐标
            pos_y: 窗口Y坐标
        """
        self.width_ = width
        self.height_ = height
        self.window_name_ = window_name
        self.pos_x_ = pos_x
        self.pos_y_ = pos_y
        self.hwnd_ = None
        self.hInstance_ = None
        self._InitWindow()
    
    def _InitWindow(self):
        """初始化 Windows 透明窗口"""
        try:
            import win32gui
            import win32con
            import win32api
            
            self.hInstance_ = win32api.GetModuleHandle(None)
            className = "TransparentOverlayClass"
            
            # 注册窗口类
            wndClass = win32gui.WNDCLASS()
            wndClass.lpfnWndProc = lambda hWnd, msg, wParam, lParam: 0
            wndClass.lpszClassName = className
            wndClass.hInstance = self.hInstance_
            
            try:
                win32gui.RegisterClass(wndClass)
            except Exception:
                # 类可能已注册，忽略错误
                pass
            
            # 创建窗口
            self.hwnd_ = win32gui.CreateWindowEx(
                win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST,
                className,
                self.window_name_,
                win32con.WS_POPUP,
                self.pos_x_,
                self.pos_y_,
                self.width_,
                self.height_,
                0,
                0,
                self.hInstance_,
                None
            )
            
            # 设置窗口位置（置顶）
            win32gui.SetWindowPos(
                self.hwnd_,
                win32con.HWND_TOPMOST,
                self.pos_x_,
                self.pos_y_,
                self.width_,
                self.height_,
                win32con.SWP_SHOWWINDOW
            )
            
            win32gui.ShowWindow(self.hwnd_, win32con.SW_SHOW)
            logger.info(f"透明浮窗初始化成功: {self.width_}x{self.height_} @ ({self.pos_x_}, {self.pos_y_})")
            
        except ImportError:
            logger.error("pywin32 未安装，请运行: pip install pywin32")
            raise
        except Exception as e:
            logger.error(f"透明浮窗初始化失败: {e}")
            raise
    
    def Draw(self, img: np.ndarray):
        """
        在透明窗口上绘制图像
        
        Args:
            img: BGR 格式的 numpy 数组
        """
        if self.hwnd_ is None:
            return
        
        try:
            import win32gui
            import win32con
            from PIL import Image
            
            # 调整图像大小
            if img.shape[0] != self.height_ or img.shape[1] != self.width_:
                img = cv2.resize(img, (self.width_, self.height_))
            
            # 转换为 BGRA 格式（添加 alpha 通道）
            if img.shape[2] == 3:
                bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            else:
                bgra = img.copy()
            
            # 转换为 PIL Image
            img_pil = Image.fromarray(bgra)
            
            # 获取设备上下文
            hdc = win32gui.GetDC(0)
            
            # 更新分层窗口
            win32gui.UpdateLayeredWindow(
                self.hwnd_,
                hdc,
                (self.pos_x_, self.pos_y_),
                (self.width_, self.height_),
                img_pil.tobytes(),
                (0, 0, self.width_, self.height_),
                0,
                (255, win32con.AC_SRC_ALPHA)
            )
            
            win32gui.ReleaseDC(0, hdc)
            
        except ImportError:
            logger.error("PIL 未安装，请运行: pip install Pillow")
        except Exception as e:
            logger.error(f"绘制失败: {e}")
    
    def DrawPath(self, minimap: np.ndarray, path: List[Tuple[int, int]], 
                 scale: float = 1.0):
        """
        在小地图上绘制路径
        
        Args:
            minimap: 小地图图像（BGR格式）
            path: 路径坐标列表 [(x, y), ...]
            scale: 缩放因子
        """
        overlay_img = minimap.copy()
        
        if len(path) > 1:
            # 绘制路径线条
            for i in range(len(path) - 1):
                pt1 = (int(path[i][0] * scale), int(path[i][1] * scale))
                pt2 = (int(path[i+1][0] * scale), int(path[i+1][1] * scale))
                cv2.line(overlay_img, pt1, pt2, (0, 255, 0), 2)
            
            # 绘制路径点
            for pt in path:
                center = (int(pt[0] * scale), int(pt[1] * scale))
                cv2.circle(overlay_img, center, 3, (0, 255, 0), -1)
        
        self.Draw(overlay_img)
    
    def SetPosition(self, x: int, y: int):
        """
        设置窗口位置
        
        Args:
            x: X坐标
            y: Y坐标
        """
        self.pos_x_ = x
        self.pos_y_ = y
        
        if self.hwnd_ is not None:
            try:
                import win32gui
                import win32con
                win32gui.SetWindowPos(
                    self.hwnd_,
                    win32con.HWND_TOPMOST,
                    x,
                    y,
                    self.width_,
                    self.height_,
                    win32con.SWP_SHOWWINDOW
                )
            except Exception as e:
                logger.error(f"设置窗口位置失败: {e}")
    
    def Close(self):
        """关闭窗口"""
        if self.hwnd_ is not None:
            try:
                import win32gui
                win32gui.DestroyWindow(self.hwnd_)
                self.hwnd_ = None
                logger.info("透明浮窗已关闭")
            except Exception as e:
                logger.error(f"关闭窗口失败: {e}")

