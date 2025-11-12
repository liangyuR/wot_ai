#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化调试工具：显示小地图检测、路径规划等结果
"""

from typing import List, Tuple, Optional
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
    from ..common.utils.logger import SetupLogger

logger = SetupLogger(__name__)


class VisualDebugger:
    """可视化调试器"""
    
    def __init__(self, enabled: bool = True):
        """
        初始化可视化调试器
        
        Args:
            enabled: 是否启用可视化
        """
        self.enabled_ = enabled
    
    def DrawDetections(self, minimap: np.ndarray, detections: dict, 
                      minimap_size: Tuple[int, int]) -> np.ndarray:
        """
        在小地图上绘制检测结果
        
        Args:
            minimap: 小地图图像
            detections: 检测结果字典
            minimap_size: 小地图尺寸 (width, height)
        
        Returns:
            绘制了检测框的图像
        """
        if not self.enabled_:
            return minimap
        
        debug_frame = minimap.copy()
        
        # 绘制自身位置
        if detections.get('self_pos'):
            pos = detections['self_pos']
            cv2.circle(debug_frame, (int(pos[0]), int(pos[1])), 10, (0, 255, 0), -1)
            cv2.putText(debug_frame, "Self", (int(pos[0]) + 15, int(pos[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制旗帜位置
        if detections.get('flag_pos'):
            pos = detections['flag_pos']
            cv2.circle(debug_frame, (int(pos[0]), int(pos[1])), 10, (0, 0, 255), -1)
            cv2.putText(debug_frame, "Flag", (int(pos[0]) + 15, int(pos[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制障碍物
        for obstacle in detections.get('obstacles', []):
            x1, y1, x2, y2 = map(int, obstacle)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 绘制道路
        for road in detections.get('roads', []):
            x1, y1, x2, y2 = map(int, road)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        return debug_frame
    
    def DrawPath(self, grid: np.ndarray, path: List[Tuple[int, int]], 
                start: Optional[Tuple[int, int]], goal: Optional[Tuple[int, int]]) -> np.ndarray:
        """
        在栅格地图上绘制路径
        
        Args:
            grid: 栅格地图
            path: 路径坐标列表
            start: 起点
            goal: 终点
        
        Returns:
            绘制了路径的图像（彩色）
        """
        if not self.enabled_:
            return grid
        
        # 转换为彩色图像
        grid_image = cv2.cvtColor(grid.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        
        # 绘制起点
        if start:
            cv2.circle(grid_image, start, 5, (0, 255, 0), -1)
            cv2.putText(grid_image, "Start", (start[0] + 10, start[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制终点
        if goal:
            cv2.circle(grid_image, goal, 5, (0, 0, 255), -1)
            cv2.putText(grid_image, "Goal", (goal[0] + 10, goal[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制路径
        if path:
            for i in range(len(path) - 1):
                pt1 = path[i]
                pt2 = path[i + 1]
                cv2.line(grid_image, pt1, pt2, (255, 0, 255), 2)
            
            # 绘制路径点
            for pt in path:
                cv2.circle(grid_image, pt, 2, (255, 0, 255), -1)
        
        return grid_image
    
    def Show(self, title: str, image: np.ndarray, wait_key: int = 1):
        """
        显示图像
        
        Args:
            title: 窗口标题
            image: 图像
            wait_key: 等待按键（0=无限等待，1=1ms）
        """
        if not self.enabled_:
            return
        
        try:
            cv2.imshow(title, image)
            cv2.waitKey(wait_key)
        except Exception as e:
            logger.error(f"显示图像失败: {e}")
    
    def CloseAll(self):
        """关闭所有窗口"""
        if self.enabled_:
            cv2.destroyAllWindows()

