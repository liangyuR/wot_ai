#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径规划主控制器：整合所有子模块，提供统一的运行接口
"""

from pathlib import Path
from typing import Optional, Callable
import threading
import time
import yaml
import numpy as np

try:
    from path_planning.core.minimap_detector import MinimapDetector
    from path_planning.core.map_modeler import MapModeler
    from path_planning.core.path_planner import AStarPlanner
    from path_planning.core.navigation_executor import NavigationExecutor
    from path_planning.service.capture_service import CaptureService
    from path_planning.service.control_service import ControlService
    from path_planning.utils.logger import SetupLogger
except ImportError:
    # 相对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.minimap_detector import MinimapDetector
    from core.map_modeler import MapModeler
    from core.path_planner import AStarPlanner
    from core.navigation_executor import NavigationExecutor
    from service.capture_service import CaptureService
    from service.control_service import ControlService
    from utils.logger import SetupLogger

logger = SetupLogger(__name__)


class PathPlanningController:
    """路径规划控制器"""
    
    def __init__(self, config_path: Path):
        """
        初始化路径规划控制器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path_ = config_path
        self.config_ = {}
        self.running_ = False
        self.thread_ = None
        
        # 子模块
        self.capture_service_ = None
        self.minimap_detector_ = None
        self.map_modeler_ = None
        self.path_planner_ = None
        self.control_service_ = None
        self.navigation_executor_ = None
        
        # 回调函数
        self.status_callback_ = None
        self.stats_callback_ = None
        
        # 统计信息
        self.stats_ = {
            'frame_count': 0,
            'detection_count': 0,
            'path_planning_count': 0,
            'navigation_count': 0,
            'fps': 0.0
        }
        
        # 加载配置
        self.LoadConfig()
    
    def LoadConfig(self) -> bool:
        """
        加载配置文件
        
        Returns:
            是否加载成功
        """
        try:
            if not self.config_path_.exists():
                logger.warning(f"配置文件不存在: {self.config_path_}，使用默认配置")
                self.config_ = self.GetDefaultConfig()
                return True
            
            with open(self.config_path_, 'r', encoding='utf-8') as f:
                self.config_ = yaml.safe_load(f) or {}
            
            # 合并默认配置
            default_config = self.GetDefaultConfig()
            self.config_ = self._MergeConfig(default_config, self.config_)
            
            logger.info(f"配置加载成功: {self.config_path_}")
            return True
        except Exception as e:
            logger.error(f"加载配置失败: {e}，使用默认配置")
            self.config_ = self.GetDefaultConfig()
            return False
    
    def GetDefaultConfig(self) -> dict:
        """获取默认配置"""
        return {
            'minimap': {
                'region': {
                    'x': 1600,
                    'y': 800,
                    'width': 320,
                    'height': 320
                },
                'grid_size': [256, 256],
                'scale_factor': 1.0
            },
            'model': {
                'path': 'train/model/minimap_yolo.pt',
                'classes': 6,
                'conf_threshold': 0.25
            },
            'navigation': {
                'mouse_sensitivity': 1.0,
                'calibration_factor': 150.0,
                'move_speed': 1.0,
                'rotation_smooth': 0.3
            },
            'planner': {
                'enable_smoothing': True,
                'smooth_weight': 0.3
            },
            'logging': {
                'level': 'INFO',
                'save_video': False
            }
        }
    
    def _MergeConfig(self, default: dict, override: dict) -> dict:
        """深度合并配置字典"""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._MergeConfig(result[key], value)
            else:
                result[key] = value
        
        return result
    
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
            
            self.minimap_detector_ = MinimapDetector(
                model_path=model_path,
                minimap_region=minimap_region,
                base_dir=Path(__file__).resolve().parent.parent.parent
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
            self.thread_.join(timeout=2.0)
        
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
    
    def SetStatusCallback(self, callback: Optional[Callable[[str], None]]):
        """
        设置状态更新回调
        
        Args:
            callback: 回调函数，参数为状态字符串
        """
        self.status_callback_ = callback
    
    def SetStatsCallback(self, callback: Optional[Callable[[dict], None]]):
        """
        设置统计信息更新回调
        
        Args:
            callback: 回调函数，参数为统计信息字典
        """
        self.stats_callback_ = callback
    
    def OnStatusUpdate(self, status: str):
        """触发状态更新回调"""
        if self.status_callback_:
            try:
                self.status_callback_(status)
            except Exception as e:
                logger.error(f"状态回调失败: {e}")
    
    def OnStatsUpdate(self):
        """触发统计信息更新回调"""
        if self.stats_callback_:
            try:
                self.stats_callback_(self.stats_)
            except Exception as e:
                logger.error(f"统计回调失败: {e}")
    
    def Run(self):
        """主循环"""
        logger.info("路径规划主循环开始")
        
        last_time = time.time()
        fps_interval = 1.0
        fps_count = 0
        fps_start_time = time.time()
        
        while self.running_:
            try:
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time
                
                # 计算FPS
                fps_count += 1
                if current_time - fps_start_time >= fps_interval:
                    self.stats_['fps'] = fps_count / (current_time - fps_start_time)
                    fps_count = 0
                    fps_start_time = current_time
                    self.OnStatsUpdate()
                
                # 1. 截取屏幕
                frame = self.capture_service_.Capture()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                self.stats_['frame_count'] += 1
                
                # 2. 检测小地图
                detections = self.minimap_detector_.Detect(
                    frame,
                    confidence_threshold=self.config_.get('model', {}).get('conf_threshold', 0.25)
                )
                
                if detections.get('self_pos') is None or detections.get('flag_pos') is None:
                    logger.debug("未检测到自身位置或目标旗帜，跳过本帧")
                    time.sleep(0.1)
                    continue
                
                self.stats_['detection_count'] += 1
                
                # 3. 构建地图模型
                minimap_size = (
                    self.config_.get('minimap', {}).get('region', {}).get('width', 320),
                    self.config_.get('minimap', {}).get('region', {}).get('height', 320)
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
                
                self.stats_['path_planning_count'] += 1
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
                        
                        self.stats_['navigation_count'] += 1
                
                # 控制循环频率
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                time.sleep(0.1)
        
        logger.info("路径规划主循环结束")

