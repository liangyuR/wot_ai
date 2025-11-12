#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时导航监控主循环：整合所有模块，实现实时小地图监控与路径规划

职责：用于可视化监控场景，实时显示路径规划结果，不执行实际控制
使用场景：调试、可视化、监控路径规划效果
与PathPlanningController的区别：
- NavigationMonitor：仅监控和显示，不执行控制操作
- PathPlanningController：完整的路径规划+控制执行，包含游戏操作
"""

# 标准库导入
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import threading

# 第三方库导入
import numpy as np

# 本地模块导入
from ..common.imports import GetLogger, ImportNavigationModule
from ..common.constants import (
    DEFAULT_GRID_SIZE,
    DEFAULT_OVERLAY_ALPHA,
    SMOOTH_WEIGHT_DEFAULT,
    THREAD_JOIN_TIMEOUT
)
from ..common.exceptions import ModelLoadError, InitializationError

logger = GetLogger()(__name__)

MinimapDetector = ImportNavigationModule('core.minimap_detector', 'MinimapDetector')
if MinimapDetector is None:
    from ..core.minimap_detector import MinimapDetector

MapModeler = ImportNavigationModule('core.map_modeler', 'MapModeler')
if MapModeler is None:
    from ..core.map_modeler import MapModeler

AStarPlanner = ImportNavigationModule('core.path_planner', 'AStarPlanner')
if AStarPlanner is None:
    from ..core.path_planner import AStarPlanner

# 导入运行时模块
from ..capture.vision_watcher import VisionWatcher
from ..ui.transparent_overlay import TransparentOverlay
from ..common.performance_profile import PerformanceProfile


class NavigationMonitor:
    """
    实时导航监控器：主循环逻辑
    
    职责：用于可视化监控场景，实时显示路径规划结果，不执行实际控制
    
    与PathPlanningController的区别：
    - NavigationMonitor：仅监控和显示，不执行控制操作，适合调试和可视化
    - PathPlanningController：完整的路径规划+控制执行，包含游戏操作，适合实际导航
    
    示例:
        ```python
        monitor = NavigationMonitor(
            model_path="path/to/model.pt",
            minimap_region={'x': 1600, 'y': 800, 'width': 320, 'height': 320},
            performance="medium"
        )
        monitor.Start()  # 后台启动
        # ... 运行中 ...
        monitor.Stop()   # 停止
        ```
    """
    
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
        try:
            if not self.detector_.LoadModel():
                raise ModelLoadError("YOLO 模型加载失败")
        except ModelLoadError:
            raise
        except Exception as e:
            raise InitializationError(f"初始化检测器失败: {e}") from e
        
        # 地图建模器
        grid_size = DEFAULT_GRID_SIZE
        self.modeler_ = MapModeler(grid_size, scale_factor=self.profile_.GetScale())
        
        # 路径规划器
        self.planner_ = AStarPlanner(enable_smoothing=True, smooth_weight=SMOOTH_WEIGHT_DEFAULT)
        
        # 屏幕捕获器
        self.watcher_ = VisionWatcher(minimap_region, self.profile_.GetFps())
        
        # 透明浮窗（位置与小地图区域对齐）
        self.overlay_ = TransparentOverlay(
            width=minimap_region['width'],
            height=minimap_region['height'],
            pos_x=minimap_region['x'],
            pos_y=minimap_region['y'],
            fps=int(self.profile_.GetFps()),
            alpha=DEFAULT_OVERLAY_ALPHA
        )
        
        # 运行状态
        self.is_running_ = False
        self.thread_ = None
        self.stop_flag_ = False
    
    def Start(self):
        """启动监控（后台线程）"""
        if self.is_running_:
            logger.warning("监控已在运行")
            return
        
        self.is_running_ = True
        self.stop_flag_ = False
        self.thread_ = threading.Thread(target=self.Run_, daemon=True)
        self.thread_.start()
        logger.info("导航监控器已启动")
    
    def Run_(self):
        """主循环（后台线程）"""
        logger.info("导航监控主循环开始")
        
        # 启动透明浮窗
        self.overlay_.StartOverlay_(self.DrawCallback_)
        
        try:
            for frame in self.watcher_.Stream():
                if self.stop_flag_:
                    break
                
                self.ProcessFrame_(frame)
        finally:
            self.overlay_.Stop()
            logger.info("导航监控主循环结束")
    
    def ProcessFrame_(self, frame: np.ndarray):
        """处理一帧画面"""
        # 提取小地图
        minimap = self.detector_.ExtractMinimap(frame)
        if minimap is None:
            return
        
        # 检测小地图元素（返回元组：self_pos, angle, flag_pos）
        # 注意：DetectMinimap_() 现在接收 minimap 而不是 frame
        self_pos, angle, flag_pos = self.detector_.DetectMinimap_(minimap)
        
        # 转换为字典格式（兼容 MapModeler）
        detections = {
            'self_pos': self_pos,
            'flag_pos': flag_pos,
            'angle': angle,
            'obstacles': [],  # 地形分析得到的障碍物
            'roads': []
        }
        
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
        self.Display_(minimap, path, detections)
    
    def Display_(self, minimap: np.ndarray, path: List[Tuple[int, int]], 
                 detections: Dict) -> None:
        """
        在小地图上显示路径和检测结果
        
        Args:
            minimap: 小地图图像
            path: 路径坐标列表
            detections: 检测结果字典
        """
        # 更新透明浮窗的绘制内容
        # 这里通过回调函数传递给 overlay
        pass
    
    def DrawCallback_(self, drawlist_id, overlay_manager):
        """绘制回调函数（由 TransparentOverlay 调用）"""
        # 这里可以绘制路径、检测结果等
        # 具体实现依赖于 overlay_manager 的绘制API
        pass
    
    def Stop(self):
        """停止监控"""
        if not self.is_running_:
            return
        
        self.stop_flag_ = True
        if self.thread_ and self.thread_.is_alive():
            self.thread_.join(timeout=THREAD_JOIN_TIMEOUT)
        logger.info("导航监控器已停止")
    
    def SetPerformanceMode(self, mode: str):
        """
        设置性能模式
        
        Args:
            mode: 性能模式 ("high", "medium", "low")
        """
        self.profile_.SetMode(mode)
        self.watcher_.SetFps(self.profile_.GetFps())
        logger.info(f"性能模式已切换为: {mode}")

