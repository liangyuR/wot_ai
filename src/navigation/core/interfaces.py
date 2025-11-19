#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心接口定义：定义导航模块的核心抽象接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, TypedDict, Protocol
import numpy as np


class DetectionResult(TypedDict, total=False):
    """检测结果字典类型定义"""
    self_pos: Optional[Tuple[float, float]]  # 我方位置 (x, y)
    flag_pos: Optional[Tuple[float, float]]  # 敌方基地位置 (x, y)
    angle: Optional[float]  # 朝向角度（度）
    obstacle_mask: Optional[np.ndarray]  # 障碍物掩码（0/1）
    obstacles: List[List[int]]  # 障碍物边界框列表 [[x1, y1, x2, y2], ...]
    roads: List  # 道路列表（向后兼容）


class StatusCallback(Protocol):
    """状态更新回调函数协议"""
    def __call__(self, status: str) -> None:
        """状态更新回调"""
        ...


class StatsCallback(Protocol):
    """统计信息更新回调函数协议"""
    def __call__(self, stats: Dict) -> None:
        """统计信息更新回调"""
        ...


class IDetector(ABC):
    """检测器接口：定义小地图检测器的标准接口"""
    
    @abstractmethod
    def LoadModel(self) -> bool:
        """
        加载检测模型
        
        Returns:
            是否加载成功
        """
        pass
    
    @abstractmethod
    def ExtractMinimap(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从屏幕帧中提取小地图区域
        
        Args:
            frame: 屏幕帧（BGR格式）
        
        Returns:
            小地图图像，失败返回None
        """
        pass
    
    @abstractmethod
    def Detect(self, frame: np.ndarray, confidence_threshold: Optional[float] = None) -> DetectionResult:
        """
        检测小地图元素
        
        Args:
            frame: 完整屏幕帧（BGR格式）
            confidence_threshold: 置信度阈值（可选）
        
        Returns:
            检测结果字典
        """
        pass


class IModeler(ABC):
    """地图建模器接口：定义地图建模器的标准接口"""
    
    @abstractmethod
    def BuildGrid(self, detections: DetectionResult, minimap_size: Tuple[int, int]) -> np.ndarray:
        """
        构建栅格地图
        
        Args:
            detections: 检测结果字典
            minimap_size: 小地图实际尺寸 (width, height)
        
        Returns:
            栅格地图（0=可通行，1=不可通行）
        """
        pass
    
    @abstractmethod
    def GetStartPos(self) -> Optional[Tuple[int, int]]:
        """
        获取起点（我方位置）
        
        Returns:
            起点坐标 (x, y)，如果未检测到则返回 None
        """
        pass
    
    @abstractmethod
    def GetGoalPos(self) -> Optional[Tuple[int, int]]:
        """
        获取终点（旗帜位置）
        
        Returns:
            终点坐标 (x, y)，如果未检测到则返回 None
        """
        pass
    
    @abstractmethod
    def GetGrid(self) -> Optional[np.ndarray]:
        """
        获取栅格地图
        
        Returns:
            栅格地图数组，如果未构建则返回 None
        """
        pass


class IPlanner(ABC):
    """路径规划器接口：定义路径规划器的标准接口"""
    
    @abstractmethod
    def Plan(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        规划路径
        
        Args:
            grid: 栅格地图（0=可通行，1=不可通行）
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
        
        Returns:
            路径坐标列表，如果无法到达则返回空列表
        """
        pass

