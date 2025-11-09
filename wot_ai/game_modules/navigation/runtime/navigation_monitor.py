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
        if not self.detector_.load_model():
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
            pos_y=minimap_region['y'],
            fps=int(self.profile_.GetFps()),
            alpha=180  # 半透明
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
        # 更新置信度阈值
        self.detector_.conf_threshold_ = self.profile_.GetConfidence()
        
        # 提取小地图区域
        minimap = self.detector_.extract_minimap(frame)
        if minimap is None:
            return

        # 检测小地图元素（返回元组：self_pos, angle, flag_pos）
        # 注意：detect() 现在接收 minimap 而不是 frame
        self_pos, angle, flag_pos = self.detector_.detect(minimap)
        
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
        # 计算小地图尺寸
        minimap_size = (minimap.shape[1], minimap.shape[0])
        
        # 使用新的 DrawPath 接口
        self.overlay_.DrawPath(
            minimap=minimap,
            path=path,
            detections=detections,
            minimap_size=minimap_size,
            grid_size=self.modeler_.grid_size_
        )
    
    def _MaskToBoundingBoxes(self, mask: np.ndarray, min_area: int = 50) -> List:
        """
        将障碍物掩码转换为边界框列表
        
        Args:
            mask: 二值掩码（1=障碍物，0=可通行）
            min_area: 最小区域面积（过滤小噪声）
        
        Returns:
            边界框列表 [[x1, y1, x2, y2], ...]
        """
        # 找到障碍物轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            obstacles.append([x, y, x + w, y + h])  # [x1, y1, x2, y2] 格式
        
        return obstacles
    
    def IsRunning(self) -> bool:
        """
        检查监控器是否正在运行
        
        Returns:
            是否正在运行
        """
        return self.is_running_

