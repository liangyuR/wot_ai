#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航执行模块：将路径转换为游戏操作
"""

from typing import List, Tuple, Optional
import time
import math
# 统一导入机制
from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple
setup_python_path()

ControlService, _ = try_import_multiple([
    'wot_ai.game_modules.navigation.service.control_service',
    'game_modules.navigation.service.control_service',
    'navigation.service.control_service'
])
if ControlService is None:
    from ..service.control_service import ControlService

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


class NavigationExecutor:
    """导航执行器：将路径转换为游戏操作"""
    
    def __init__(self, control_service: ControlService, move_speed: float = 1.0, rotation_smooth: float = 0.3):
        """
        初始化导航执行器
        
        Args:
            control_service: 控制服务实例
            move_speed: 移动速度系数
            rotation_smooth: 转向平滑系数
        """
        self.control_service_ = control_service
        self.move_speed_ = move_speed
        self.rotation_smooth_ = rotation_smooth
        self.current_smoothed_angle_ = 0.0
    
    def ExecutePath(self, path: List[Tuple[int, int]], current_pos: Optional[Tuple[float, float]], 
                    update_interval: float = 0.1) -> None:
        """
        执行路径
        
        Args:
            path: 路径坐标列表
            current_pos: 当前位置（小地图坐标，可为None）
            update_interval: 更新间隔（秒）
        """
        if not path:
            logger.warning("路径为空，无法执行")
            return
        
        if len(path) == 1:
            logger.info("已到达目标位置")
            return
        
        # 执行路径的每个点
        for i, waypoint in enumerate(path[1:], 1):
            logger.debug(f"前往路径点 {i}/{len(path)-1}: {waypoint}")
            
            # 如果当前位置未知，直接执行
            if current_pos is None:
                # 转向目标方向
                if i > 0:
                    prev_point = path[i-1]
                    self.RotateToward(waypoint, prev_point)
                
                # 前进
                self.MoveForward(0.5 * self.move_speed_)
                time.sleep(update_interval)
            else:
                # 转向目标
                self.RotateToward(waypoint, current_pos)
                
                # 计算距离
                distance = math.sqrt(
                    (waypoint[0] - current_pos[0]) ** 2 + 
                    (waypoint[1] - current_pos[1]) ** 2
                )
                
                # 根据距离计算前进时间
                duration = min(distance / 100.0 * self.move_speed_, 2.0)  # 最大2秒
                self.MoveForward(duration)
                
                # 更新当前位置（假设已经到达）
                current_pos = waypoint
                time.sleep(update_interval)
    
    def RotateToward(self, target_pos: Tuple[float, float], current_pos: Tuple[float, float]) -> None:
        """
        转向目标位置
        
        Args:
            target_pos: 目标位置 (x, y)
            current_pos: 当前位置 (x, y)
        """
        self.control_service_.RotateToward(target_pos, current_pos)
    
    def MoveForward(self, duration: float) -> None:
        """
        前进
        
        Args:
            duration: 持续时间（秒）
        """
        self.control_service_.MoveForward(duration)
    
    def Stop(self) -> None:
        """停止移动"""
        self.control_service_.Stop()

