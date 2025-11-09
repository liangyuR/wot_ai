#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径规划主控制器：整合所有子模块，提供统一的运行接口

职责：完整的路径规划+控制执行系统，包含游戏操作
使用场景：实际游戏导航，需要执行控制操作的场景
与NavigationMonitor的区别：
- PathPlanningController：完整的路径规划+控制执行，包含游戏操作，适合实际导航
- NavigationMonitor：仅监控和显示，不执行控制操作，适合调试和可视化
"""

# 标准库导入
from pathlib import Path
from typing import Optional
import threading
import time

# 第三方库导入
import numpy as np

# 本地模块导入
from wot_ai.utils.paths import resolve_path
from ..common.imports import GetLogger, ImportNavigationModule
from ..common.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MINIMAP_WIDTH,
    DEFAULT_MINIMAP_HEIGHT,
    THREAD_JOIN_TIMEOUT
)
from ..core.interfaces import StatusCallback, StatsCallback
from .config_manager import ConfigManager
from .stats_collector import StatsCollector

# 导入依赖模块
MinimapDetector = ImportNavigationModule('core.minimap_detector', 'MinimapDetector')
if MinimapDetector is None:
    from .core.minimap_detector import MinimapDetector

MapModeler = ImportNavigationModule('core.map_modeler', 'MapModeler')
if MapModeler is None:
    from .core.map_modeler import MapModeler

AStarPlanner = ImportNavigationModule('core.path_planner', 'AStarPlanner')
if AStarPlanner is None:
    from .core.path_planner import AStarPlanner

NavigationExecutor = ImportNavigationModule('core.navigation_executor', 'NavigationExecutor')
if NavigationExecutor is None:
    from .core.navigation_executor import NavigationExecutor

CaptureService = ImportNavigationModule('service.capture_service', 'CaptureService')
if CaptureService is None:
    from .service.capture_service import CaptureService

ControlService = ImportNavigationModule('service.control_service', 'ControlService')
if ControlService is None:
    from .service.control_service import ControlService

logger = GetLogger()(__name__)


class PathPlanningController:
    """
    路径规划控制器
    
    职责：完整的路径规划+控制执行系统，包含游戏操作
    
    与NavigationMonitor的区别：
    - PathPlanningController：完整的路径规划+控制执行，包含游戏操作，适合实际导航
    - NavigationMonitor：仅监控和显示，不执行控制操作，适合调试和可视化
    
    示例:
        ```python
        controller = PathPlanningController(config_path=Path("config.yaml"))
        controller.SetStatusCallback(lambda status: print(f"Status: {status}"))
        controller.Start()  # 启动路径规划和控制
        # ... 运行中 ...
        controller.Stop()   # 停止
        ```
    """
    
    def __init__(self, config_path: Path):
        """
        初始化路径规划控制器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path_ = config_path
        self.running_ = False
        self.thread_ = None
        
        # 配置管理器
        self.config_manager_ = ConfigManager(config_path)
        self.config_manager_.LoadConfig()
        self.config_ = self.config_manager_.GetConfig()
        
        # 统计收集器
        self.stats_collector_ = StatsCollector()
        
        # 子模块
        self.capture_service_ = None
        self.minimap_detector_ = None
        self.map_modeler_ = None
        self.path_planner_ = None
        self.control_service_ = None
        self.navigation_executor_ = None
        
        # 回调函数
        self.status_callback_: Optional[StatusCallback] = None
        self.stats_callback_: Optional[StatsCallback] = None
    
    
    def Initialize(self) -> bool:
        """
        初始化所有子模块
        
        Returns:
            是否初始化成功
        """
        try:
            # 初始化屏幕捕获服务
            self.capture_service_ = CaptureService(monitor_index=1)
            
            # 初始化小地图检测器
            model_config = self.config_.get('model', {})
            minimap_config = self.config_.get('minimap', {})
            model_path = model_config.get('path', 'train/model/minimap_yolo.pt')
            minimap_region = minimap_config.get('region', {})
            
            # 使用统一路径解析
            resolved_model_path = resolve_path(model_path)
            self.minimap_detector_ = MinimapDetector(
                model_path=str(resolved_model_path),
                minimap_region=minimap_region,
                base_dir=None  # 使用统一路径管理，不需要 base_dir
            )
            
            if not self.minimap_detector_.LoadModel():
                logger.error("小地图检测模型加载失败")
                return False
            
            # 初始化地图建模器
            grid_size = tuple(minimap_config.get('grid_size', [256, 256]))
            scale_factor = minimap_config.get('scale_factor', 1.0)
            self.map_modeler_ = MapModeler(grid_size=grid_size, scale_factor=scale_factor)
            
            # 初始化路径规划器
            planner_config = self.config_.get('planner', {})
            self.path_planner_ = AStarPlanner(
                enable_smoothing=planner_config.get('enable_smoothing', True),
                smooth_weight=planner_config.get('smooth_weight', 0.3)
            )
            
            # 初始化控制服务
            nav_config = self.config_.get('navigation', {})
            self.control_service_ = ControlService(
                mouse_sensitivity=nav_config.get('mouse_sensitivity', 1.0),
                calibration_factor=nav_config.get('calibration_factor', 150.0)
            )
            
            # 初始化导航执行器
            self.navigation_executor_ = NavigationExecutor(
                control_service=self.control_service_,
                move_speed=nav_config.get('move_speed', 1.0),
                rotation_smooth=nav_config.get('rotation_smooth', 0.3)
            )
            
            logger.info("所有子模块初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def Start(self) -> bool:
        """
        启动路径规划
        
        Returns:
            是否启动成功
        """
        if self.running_:
            logger.warning("路径规划已在运行")
            return False
        
        if not self.Initialize():
            logger.error("初始化失败，无法启动")
            return False
        
        self.running_ = True
        self.thread_ = threading.Thread(target=self.Run, daemon=True)
        self.thread_.start()
        
        logger.info("路径规划已启动")
        self.OnStatusUpdate("运行中")
        return True
    
    def Stop(self):
        """停止路径规划"""
        if not self.running_:
            return
        
        self.running_ = False
        
        if self.thread_ and self.thread_.is_alive():
            self.thread_.join(timeout=THREAD_JOIN_TIMEOUT)
        
        # 停止移动
        if self.navigation_executor_:
            self.navigation_executor_.Stop()
        
        logger.info("路径规划已停止")
        self.OnStatusUpdate("已停止")
    
    def IsRunning(self) -> bool:
        """
        检查是否正在运行
        
        Returns:
            是否正在运行
        """
        return self.running_
    
    def SetStatusCallback(self, callback: Optional[StatusCallback]) -> None:
        """
        设置状态更新回调
        
        Args:
            callback: 回调函数，参数为状态字符串
        """
        self.status_callback_ = callback
    
    def SetStatsCallback(self, callback: Optional[StatsCallback]) -> None:
        """
        设置统计信息更新回调
        
        Args:
            callback: 回调函数，参数为统计信息字典
        """
        self.stats_callback_ = callback
    
    def OnStatusUpdate(self, status: str) -> None:
        """触发状态更新回调"""
        if self.status_callback_:
            try:
                self.status_callback_(status)
            except Exception as e:
                logger.error(f"状态回调失败: {e}")
    
    def OnStatsUpdate(self) -> None:
        """触发统计信息更新回调"""
        if self.stats_callback_:
            try:
                self.stats_callback_(self.stats_collector_.GetStats())
            except Exception as e:
                logger.error(f"统计回调失败: {e}")
    
    def Run(self):
        """主循环"""
        logger.info("路径规划主循环开始")
        
        while self.running_:
            try:
                # 更新FPS（如果达到更新间隔）
                if self.stats_collector_.UpdateFps():
                    self.OnStatsUpdate()
                
                # 1. 截取屏幕
                frame = self.capture_service_.Capture()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                self.stats_collector_.UpdateFrameCount()
                
                # 2. 检测小地图
                detections = self.minimap_detector_.Detect(
                    frame,
                    confidence_threshold=self.config_.get('model', {}).get('conf_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
                )
                
                if detections.get('self_pos') is None or detections.get('flag_pos') is None:
                    logger.debug("未检测到自身位置或目标旗帜，跳过本帧")
                    time.sleep(0.1)
                    continue
                
                self.stats_collector_.UpdateDetectionCount()
                
                # 3. 构建地图模型
                minimap_size = (
                    self.config_.get('minimap', {}).get('region', {}).get('width', DEFAULT_MINIMAP_WIDTH),
                    self.config_.get('minimap', {}).get('region', {}).get('height', DEFAULT_MINIMAP_HEIGHT)
                )
                grid = self.map_modeler_.BuildGrid(detections, minimap_size)
                
                # 4. 规划路径
                start_pos = self.map_modeler_.GetStartPos()
                goal_pos = self.map_modeler_.GetGoalPos()
                
                if start_pos is None or goal_pos is None:
                    logger.warning("无法获取起点或终点")
                    time.sleep(0.1)
                    continue
                
                path = self.path_planner_.Plan(grid, start_pos, goal_pos)
                
                if not path:
                    logger.warning("无法规划路径")
                    time.sleep(0.1)
                    continue
                
                self.stats_collector_.UpdatePathPlanningCount()
                logger.info(f"路径规划成功，路径长度: {len(path)}")
                
                # 5. 执行导航（反馈式导航，执行第一个路径点）
                if len(path) > 1:
                    next_point = path[1]
                    current_pos = detections.get('self_pos')
                    
                    # 转换为小地图坐标
                    if current_pos:
                        # 执行单步导航
                        self.navigation_executor_.RotateToward(
                            (next_point[0], next_point[1]),
                            current_pos
                        )
                        
                        # 计算距离并前进
                        distance = ((next_point[0] - current_pos[0]) ** 2 + 
                                   (next_point[1] - current_pos[1]) ** 2) ** 0.5
                        duration = min(distance / 100.0 * self.config_.get('navigation', {}).get('move_speed', 1.0), 0.5)
                        self.navigation_executor_.MoveForward(duration)
                        
                        self.stats_collector_.UpdateNavigationCount()
                
                # 控制循环频率
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                time.sleep(0.1)
        
        logger.info("路径规划主循环结束")

