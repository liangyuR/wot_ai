#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航执行模块：将路径转换为游戏操作
"""

# 标准库导入
from typing import List, Tuple, Optional
import math
import time

# 本地模块导入
from ..common.constants import (
    DEFAULT_MOVE_SPEED,
    DEFAULT_ROTATION_SMOOTH,
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_MOVE_DURATION,
    MAX_MOVE_DURATION
)
from loguru import logger
from ..service.control_service import ControlService

class NavigationExecutor:
    """导航执行器：将路径转换为游戏操作"""
    
    def __init__(self, control_service: ControlService, move_speed: float = DEFAULT_MOVE_SPEED, rotation_smooth: float = DEFAULT_ROTATION_SMOOTH):
        """
        初始化导航执行器
        
        Args:
            control_service: 控制服务实例
            move_speed: 移动速度系数
            rotation_smooth: 转向平滑系数
        
        Raises:
            ValueError: 输入参数无效
        """
        if control_service is None:
            raise ValueError("control_service不能为None")
        if not isinstance(move_speed, (int, float)) or move_speed <= 0:
            raise ValueError("move_speed必须是正数")
        if not isinstance(rotation_smooth, (int, float)) or rotation_smooth < 0 or rotation_smooth > 1:
            raise ValueError("rotation_smooth必须在0-1之间")
        
        self.control_service_ = control_service
        self.move_speed_ = move_speed
        self.rotation_smooth_ = rotation_smooth
        self.current_smoothed_angle_ = 0.0
        # 路径消费状态
        self.target_point_offset_ = 5  # 目标点偏移量（选择路径上第几个点作为目标）
        self.current_target_idx_ = 0  # 当前目标点在路径中的索引
    
    def ExecutePath(self, path: List[Tuple[int, int]], current_pos: Optional[Tuple[float, float]], 
                    update_interval: float = DEFAULT_UPDATE_INTERVAL) -> None:
        """
        执行路径
        
        Args:
            path: 路径坐标列表
            current_pos: 当前位置（小地图坐标，可为None）
            update_interval: 更新间隔（秒）
        
        Raises:
            ValueError: 输入参数无效
        """
        if not path:
            logger.warning("路径为空，无法执行")
            return
        if not isinstance(path, list):
            raise ValueError("path必须是列表")
        if update_interval <= 0:
            raise ValueError("update_interval必须是正数")
        
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
                self.MoveForward(DEFAULT_MOVE_DURATION * self.move_speed_)
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
                duration = min(distance / 100.0 * self.move_speed_, MAX_MOVE_DURATION)
                self.MoveForward(duration)
                
                # 更新当前位置（假设已经到达）
                current_pos = waypoint
                time.sleep(update_interval)
    
    def RotateToward(self, target_pos: Tuple[float, float], current_pos: Tuple[float, float], current_heading: float) -> None:
        """
        转向目标位置
        
        Args:
            target_pos: 目标位置 (x, y)
            current_pos: 当前位置 (x, y)
            current_heading: 当前朝向角度（弧度）
        """
        self.control_service_.RotateToward(target_pos, current_pos, current_heading)
    
    def MoveForward(self, duration: float) -> None:
        """
        前进
        
        Args:
            duration: 持续时间（秒）
        """
        self.control_service_.MoveForward(duration)
    
    def EnsureMovingForward(self) -> None:
        """
        确保正在持续前进（按下W键）
        如果已按下则不做任何操作，避免重复按压
        """
        self.control_service_.StartForward()
    
    def StopMoving(self) -> None:
        """
        停止移动（释放W键）
        """
        self.control_service_.StopForward()
        logger.debug("停止移动")
    
    def Stop(self) -> None:
        """停止移动（释放所有按键）"""
        self.StopMoving()
        self.control_service_.Stop()

