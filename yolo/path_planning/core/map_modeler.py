#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图建模模块：将检测结果转换为二维栅格地图
"""

from typing import Dict, Tuple, Optional
import numpy as np
try:
    from path_planning.utils.logger import SetupLogger
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.logger import SetupLogger

logger = SetupLogger(__name__)


class MapModeler:
    """地图建模器：将检测结果转换为栅格地图"""
    
    def __init__(self, grid_size: Tuple[int, int], scale_factor: float = 1.0):
        """
        初始化地图建模器
        
        Args:
            grid_size: 栅格地图尺寸 (width, height)
            scale_factor: 缩放因子（用于适配不同分辨率）
        """
        self.grid_size_ = grid_size
        self.scale_factor_ = scale_factor
        self.grid_ = None
        self.start_pos_ = None
        self.goal_pos_ = None
        self.minimap_width_ = None
        self.minimap_height_ = None
    
    def BuildGrid(self, detections: Dict, minimap_size: Tuple[int, int]) -> np.ndarray:
        """
        构建栅格地图
        
        Args:
            detections: 检测结果字典
            minimap_size: 小地图实际尺寸 (width, height)
        
        Returns:
            栅格地图（0=可通行，1=不可通行）
        """
        self.minimap_width_, self.minimap_height_ = minimap_size
        
        # 初始化栅格地图（全为0，可通行）
        grid = np.zeros((self.grid_size_[1], self.grid_size_[0]), dtype=np.uint8)
        
        # 计算缩放比例
        scale_x = self.grid_size_[0] / minimap_size[0] * self.scale_factor_
        scale_y = self.grid_size_[1] / minimap_size[1] * self.scale_factor_
        
        # 标记障碍物
        for obstacle in detections.get('obstacles', []):
            x1, y1, x2, y2 = obstacle
            
            # 转换到栅格坐标
            grid_x1 = int(x1 * scale_x)
            grid_y1 = int(y1 * scale_y)
            grid_x2 = int(x2 * scale_x)
            grid_y2 = int(y2 * scale_y)
            
            # 确保在边界内
            grid_x1 = max(0, min(grid_x1, self.grid_size_[0] - 1))
            grid_y1 = max(0, min(grid_y1, self.grid_size_[1] - 1))
            grid_x2 = max(0, min(grid_x2, self.grid_size_[0] - 1))
            grid_y2 = max(0, min(grid_y2, self.grid_size_[1] - 1))
            
            # 填充障碍物区域
            grid[grid_y1:grid_y2+1, grid_x1:grid_x2+1] = 1
        
        self.grid_ = grid
        
        # 保存起点和终点
        self_pos = detections.get('self_pos')
        flag_pos = detections.get('flag_pos')
        
        if self_pos is not None:
            grid_x = int(self_pos[0] * scale_x)
            grid_y = int(self_pos[1] * scale_y)
            grid_x = max(0, min(grid_x, self.grid_size_[0] - 1))
            grid_y = max(0, min(grid_y, self.grid_size_[1] - 1))
            self.start_pos_ = (grid_x, grid_y)
        else:
            self.start_pos_ = None
        
        if flag_pos is not None:
            grid_x = int(flag_pos[0] * scale_x)
            grid_y = int(flag_pos[1] * scale_y)
            grid_x = max(0, min(grid_x, self.grid_size_[0] - 1))
            grid_y = max(0, min(grid_y, self.grid_size_[1] - 1))
            self.goal_pos_ = (grid_x, grid_y)
        else:
            self.goal_pos_ = None
        
        return grid
    
    def IsObstacle(self, x: int, y: int) -> bool:
        """
        检查指定位置是否为障碍物
        
        Args:
            x: 栅格X坐标
            y: 栅格Y坐标
        
        Returns:
            是否为障碍物
        """
        if self.grid_ is None:
            return False
        
        if x < 0 or x >= self.grid_size_[0] or y < 0 or y >= self.grid_size_[1]:
            return True  # 边界外视为障碍物
        
        return self.grid_[y, x] == 1
    
    def GetStartPos(self) -> Optional[Tuple[int, int]]:
        """
        获取起点（我方位置）
        
        Returns:
            起点坐标 (x, y)，如果未检测到则返回 None
        """
        return self.start_pos_
    
    def GetGoalPos(self) -> Optional[Tuple[int, int]]:
        """
        获取终点（旗帜位置）
        
        Returns:
            终点坐标 (x, y)，如果未检测到则返回 None
        """
        return self.goal_pos_
    
    def GetGrid(self) -> Optional[np.ndarray]:
        """
        获取栅格地图
        
        Returns:
            栅格地图数组，如果未构建则返回 None
        """
        return self.grid_

