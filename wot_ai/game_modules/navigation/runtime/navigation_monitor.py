#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时导航监控主循环：整合所有模块，实现实时小地图监控与路径规划
"""

from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np
import cv2
import threading
import time

# 统一导入机制
from wot_ai.utils.paths import setup_python_path, resolve_path
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

# 导入核心模块
from wot_ai.utils.imports import import_function

MinimapDetector = import_function([
    'wot_ai.game_modules.navigation.core.minimap_detector',
    'game_modules.navigation.core.minimap_detector',
    'navigation.core.minimap_detector'
], 'MinimapDetector')
if MinimapDetector is None:
    from ..core.minimap_detector import MinimapDetector

MapModeler = import_function([
    'wot_ai.game_modules.navigation.core.map_modeler',
    'game_modules.navigation.core.map_modeler',
    'navigation.core.map_modeler'
], 'MapModeler')
if MapModeler is None:
    from ..core.map_modeler import MapModeler

AStarPlanner = import_function([
    'wot_ai.game_modules.navigation.core.path_planner',
    'game_modules.navigation.core.path_planner',
    'navigation.core.path_planner'
], 'AStarPlanner')
if AStarPlanner is None:
    from ..core.path_planner import AStarPlanner

# 导入运行时模块
from .vision_watcher import VisionWatcher
from .transparent_overlay import TransparentOverlay
from .performance_profile import PerformanceProfile


class NavigationMonitor:
    """实时导航监控器：主循环逻辑"""
    
    def __init__(self, model_path: str, minimap_region: Dict, 
                 performance: str = "medium", base_dir: Optional[Path] = None):
        """
        初始化导航监控器
        
        Args:
            model_path: YOLO 模型路径
            minimap_region: 小地图区域配置 {'x': int, 'y': int, 'width': int, 'height': int}
            performance: 性能模式 ("high", "medium", "low")
            base_dir: 基础目录（用于解析相对路径）
        """
        # 性能配置
        self.profile_ = PerformanceProfile(performance)
        
        # 小地图检测器
        self.detector_ = MinimapDetector(model_path, minimap_region, base_dir)
        if not self.detector_.LoadModel():
            raise RuntimeError("YOLO 模型加载失败")
        
        # 地图建模器
        grid_size = (64, 64)  # 默认栅格大小
        self.modeler_ = MapModeler(grid_size, scale_factor=self.profile_.GetScale())
        
        # 路径规划器
        self.planner_ = AStarPlanner(enable_smoothing=True, smooth_weight=0.3)
        
        # 屏幕捕获器
        self.watcher_ = VisionWatcher(minimap_region, self.profile_.GetFps())
        
        # 透明浮窗（位置与小地图区域对齐）
        self.overlay_ = TransparentOverlay(
            width=minimap_region['width'],
            height=minimap_region['height'],
            pos_x=minimap_region['x'],
            pos_y=minimap_region['y']
        )
        
        # 运行状态
        self.is_running_ = False
        self.thread_ = None
        self.stop_flag_ = False
    
    def Run(self):
        """运行主循环（阻塞）"""
        self.is_running_ = True
        self.stop_flag_ = False
        
        logger.info("导航监控器开始运行")
        
        try:
            for frame in self.watcher_.Stream():
                if self.stop_flag_:
                    break
                
                self._ProcessFrame(frame)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        except Exception as e:
            logger.error(f"主循环异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running_ = False
            self.overlay_.Close()
            logger.info("导航监控器已停止")
    
    def Start(self):
        """在后台线程中启动监控"""
        if self.is_running_:
            logger.warning("监控器已在运行")
            return
        
        self.stop_flag_ = False
        self.thread_ = threading.Thread(target=self.Run, daemon=True)
        self.thread_.start()
        logger.info("导航监控器已在后台启动")
    
    def Stop(self):
        """停止监控"""
        if not self.is_running_:
            return
        
        self.stop_flag_ = True
        if self.thread_ and self.thread_.is_alive():
            self.thread_.join(timeout=2.0)
        logger.info("导航监控器已停止")
    
    def SetPerformanceMode(self, mode: str):
        """
        设置性能模式
        
        Args:
            mode: 性能模式 ("high", "medium", "low")
        """
        self.profile_.SetMode(mode)
        self.watcher_.SetFps(self.profile_.GetFps())
        self.modeler_.scale_factor_ = self.profile_.GetScale()
        logger.info(f"性能模式已切换为: {mode}")
    
    def _ProcessFrame(self, frame: np.ndarray):
        """
        处理一帧画面
        
        Args:
            frame: 屏幕帧（BGR格式）
        """
        # 检测小地图元素（Detect 内部会提取小地图）
        detections = self.detector_.Detect(
            frame, 
            confidence_threshold=self.profile_.GetConfidence()
        )
        
        # 提取小地图区域用于显示
        minimap = self.detector_.ExtractMinimap(frame)
        if minimap is None:
            return
        
        # 构建栅格地图
        minimap_size = (minimap.shape[1], minimap.shape[0])
        grid = self.modeler_.BuildGrid(detections, minimap_size)
        
        # 获取起点和终点
        start = self.modeler_.GetStartPos()
        goal = self.modeler_.GetGoalPos()
        
        # 路径规划
        path = []
        if start is not None and goal is not None:
            path = self.planner_.Plan(grid, start, goal)
        
        # 显示路径
        self._Display(minimap, path, detections)
    
    def _Display(self, minimap: np.ndarray, path: List[Tuple[int, int]], 
                 detections: Dict):
        """
        在小地图上显示路径和检测结果
        
        Args:
            minimap: 小地图图像
            path: 路径坐标列表（栅格坐标）
            detections: 检测结果
        """
        # 创建叠加图像
        overlay_img = minimap.copy()
        
        # 计算缩放比例（从栅格坐标到像素坐标）
        grid_size = self.modeler_.grid_size_
        minimap_width = minimap.shape[1]
        minimap_height = minimap.shape[0]
        scale_x = minimap_width / grid_size[0]
        scale_y = minimap_height / grid_size[1]
        
        # 绘制路径
        if len(path) > 1:
            # 转换为像素坐标
            pixel_path = []
            for pt in path:
                px = int(pt[0] * scale_x)
                py = int(pt[1] * scale_y)
                pixel_path.append((px, py))
            
            # 绘制路径线条
            for i in range(len(pixel_path) - 1):
                cv2.line(overlay_img, pixel_path[i], pixel_path[i+1], 
                        (0, 255, 0), 2)
            
            # 绘制路径点
            for pt in pixel_path:
                cv2.circle(overlay_img, pt, 3, (0, 255, 0), -1)
        
        # 绘制起点（自己位置）
        if detections.get('self_pos') is not None:
            pos = detections['self_pos']
            center = (int(pos[0]), int(pos[1]))
            cv2.circle(overlay_img, center, 5, (255, 255, 0), -1)
            cv2.circle(overlay_img, center, 8, (255, 255, 0), 2)
        
        # 绘制终点（基地位置）
        if detections.get('flag_pos') is not None:
            pos = detections['flag_pos']
            center = (int(pos[0]), int(pos[1]))
            cv2.circle(overlay_img, center, 5, (0, 0, 255), -1)
            cv2.circle(overlay_img, center, 8, (0, 0, 255), 2)
        
        # 更新透明浮窗
        self.overlay_.Draw(overlay_img)
    
    def IsRunning(self) -> bool:
        """
        检查监控器是否正在运行
        
        Returns:
            是否正在运行
        """
        return self.is_running_

