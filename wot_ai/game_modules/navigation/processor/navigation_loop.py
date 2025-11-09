#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航运行时循环：端到端的导航处理流程
"""

# 标准库导入
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 第三方库导入
import numpy as np

# 本地模块导入
from ..common.imports import GetLogger, ImportNavigationModule
from ..common.constants import (
    DEFAULT_GRID_SIZE,
    DEFAULT_MAPS_DIR,
    DEFAULT_INFLATE_PX,
    SMOOTH_WEIGHT_LOW,
    SMOOTH_WEIGHT_MEDIUM,
    SMOOTH_WEIGHT_HIGH
)
from ..common.exceptions import ModelLoadError, InitializationError

logger = GetLogger()(__name__)

# 导入核心模块
MinimapDetector = ImportNavigationModule('core.minimap_detector', 'MinimapDetector')
if MinimapDetector is None:
    from ..core.minimap_detector import MinimapDetector

MapModeler = ImportNavigationModule('core.map_modeler', 'MapModeler')
if MapModeler is None:
    from ..core.map_modeler import MapModeler

AStarPlanner = ImportNavigationModule('core.path_planner', 'AStarPlanner')
if AStarPlanner is None:
    from ..core.path_planner import AStarPlanner


class NavigationRuntime:
    """导航运行时：端到端的导航处理"""
    
    def __init__(self, map_id: str, minimap_region: Dict, model_path: str,
                 grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE, maps_dir: str = DEFAULT_MAPS_DIR,
                 inflate_px: int = DEFAULT_INFLATE_PX, performance: str = "medium", 
                 base_dir: Optional[Path] = None):
        """
        初始化导航运行时
        
        Args:
            map_id: 地图ID
            minimap_region: 小地图区域配置 {'x': int, 'y': int, 'width': int, 'height': int}
            model_path: YOLO模型路径
            grid_size: 栅格尺寸 (width, height)
            maps_dir: 地图数据目录
            inflate_px: 膨胀像素数
            performance: 性能模式 ("high", "medium", "low")
            base_dir: 基础目录（用于解析相对路径）
        """
        self.map_id_ = map_id
        self.minimap_region_ = minimap_region
        self.grid_size_ = grid_size
        
        # 初始化检测器（启用静态掩码）
        self.detector_ = MinimapDetector(
            model_path=model_path,
            minimap_region=minimap_region,
            base_dir=base_dir,
            map_id=map_id,
            maps_dir=maps_dir,
            inflate_px=inflate_px
        )
        try:
            if not self.detector_.LoadModel():
                raise ModelLoadError("YOLO 模型加载失败")
        except ModelLoadError:
            raise
        except Exception as e:
            raise InitializationError(f"初始化检测器失败: {e}") from e
        
        # 初始化地图建模器
        self.modeler_ = MapModeler(grid_size, scale_factor=1.0)
        
        # 初始化路径规划器
        smooth_weight = {
            "low": SMOOTH_WEIGHT_LOW,
            "medium": SMOOTH_WEIGHT_MEDIUM,
            "high": SMOOTH_WEIGHT_HIGH
        }.get(performance, SMOOTH_WEIGHT_MEDIUM)
        self.planner_ = AStarPlanner(enable_smoothing=True, smooth_weight=smooth_weight)
        
        if logger:
            logger.info(f"导航运行时已初始化: map_id={map_id}, grid={grid_size}, perf={performance}")
    
    def Step(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]]]:
        """
        处理一帧画面，返回栅格地图和路径
        
        Args:
            frame_bgr: 屏幕帧（BGR格式）
        
        Returns:
            (grid, path)
            - grid: 栅格地图（0=可通行，1=障碍），如果处理失败返回None
            - path: 路径坐标列表（栅格坐标），如果无法规划返回空列表
        """
        try:
            # 1. 检测小地图元素（包含静态掩码）
            detections = self.detector_.Detect(frame_bgr, confidence_threshold=self.detector_.conf_threshold_)
            
            # 2. 提取小地图用于获取尺寸
            minimap = self.detector_.ExtractMinimap(frame_bgr)
            if minimap is None:
                return None, []
            
            minimap_size = (minimap.shape[1], minimap.shape[0])
            
            # 3. 构建栅格地图（优先使用静态掩码）
            if detections.get('obstacle_mask') is not None:
                # 使用静态掩码构建栅格
                mask01 = detections['obstacle_mask']
                grid = self.modeler_.BuildGridFromMask(mask01)
            else:
                # 回退到传统方法（基于检测结果）
                grid = self.modeler_.BuildGrid(detections, minimap_size)
            
            # 4. 转换起点和终点到栅格坐标
            scale_x = self.grid_size_[0] / minimap_size[0]
            scale_y = self.grid_size_[1] / minimap_size[1]
            
            start = None
            goal = None
            
            if detections.get('self_pos'):
                sx, sy = detections['self_pos']
                start = (int(sx * scale_x), int(sy * scale_y))
                start = (max(0, min(start[0], self.grid_size_[0] - 1)),
                        max(0, min(start[1], self.grid_size_[1] - 1)))
            
            if detections.get('flag_pos'):
                gx, gy = detections['flag_pos']
                goal = (int(gx * scale_x), int(gy * scale_y))
                goal = (max(0, min(goal[0], self.grid_size_[0] - 1)),
                       max(0, min(goal[1], self.grid_size_[1] - 1)))
            
            # 5. 路径规划
            path = []
            if start is not None and goal is not None:
                path = self.planner_.Plan(grid, start, goal)
                if logger:
                    logger.debug(f"路径规划完成: start={start}, goal={goal}, path_len={len(path)}")
            
            return grid, path
            
        except Exception as e:
            if logger:
                logger.error(f"处理帧失败: {e}")
            return None, []

