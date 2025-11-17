#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径规划服务模块

封装路径规划的完整流程，整合坐标转换、A*规划、路径平滑。
"""

from typing import Optional, List, Tuple
from loguru import logger

from wot_ai.game_modules.vision.detection.minimap_detector import MinimapDetectionResult
from wot_ai.game_modules.navigation.core.coordinate_utils import world_to_grid
from wot_ai.game_modules.navigation.core.path_planner import AStarPlanner
from wot_ai.game_modules.navigation.core.planner_astar import astar_with_cost
from wot_ai.game_modules.navigation.core.path_smoothing import smooth_path_los, smooth_path
from wot_ai.game_modules.navigation.config.models import NavigationConfig


class PathPlanningService:
    """路径规划服务"""
    
    def __init__(self, config: NavigationConfig):
        """
        初始化路径规划服务
        
        Args:
            config: NavigationConfig配置对象
        """
        self.config_ = config
        self.planner_ = AStarPlanner(
            enable_smoothing=config.path_planning.enable_astar_smoothing,
            smooth_weight=config.path_planning.astar_smooth_weight
        )
        self.grid_: Optional = None
        self.cost_map_: Optional = None
        self.inflated_obstacle_: Optional = None
    
    def set_mask_data(self, grid, cost_map, inflated_obstacle) -> None:
        """
        设置掩码数据
        
        Args:
            grid: 栅格地图
            cost_map: 代价图
            inflated_obstacle: 膨胀障碍图
        """
        self.grid_ = grid
        self.cost_map_ = cost_map
        self.inflated_obstacle_ = inflated_obstacle
    
    def plan_path(
        self,
        minimap_size: Tuple[int, int],
        detections: MinimapDetectionResult
    ) -> Optional[List[Tuple[int, int]]]:
        """
        规划路径
        
        Args:
            minimap_size: 小地图尺寸 (width, height)
            detections: 检测结果
        
        Returns:
            路径坐标列表（栅格坐标），失败返回 None
        """
        if detections.self_pos is None or detections.enemy_flag_pos is None:
            logger.warning("检测结果不完整，无法规划路径")
            return None
        
        if self.grid_ is None:
            logger.error("栅格地图未初始化，请先调用 set_mask_data")
            return None
        
        try:
            grid_size = self.config_.grid.size
            
            # 将检测结果的位置转换为栅格坐标
            start = world_to_grid(detections.self_pos, minimap_size, grid_size)
            goal = world_to_grid(detections.enemy_flag_pos, minimap_size, grid_size)
            
            logger.info(f"起点（栅格坐标）: {start}, 终点（栅格坐标）: {goal}")
            
            # 使用带代价图的A*算法规划路径
            if self.cost_map_ is not None:
                # 使用新的cost_map A*
                path = astar_with_cost(self.cost_map_, start, goal)
                if not path:
                    logger.warning("cost_map A*规划失败，尝试使用传统A*")
                    path = self.planner_.Plan(self.grid_, start, goal)
                else:
                    # 路径平滑：可选使用Catmull-Rom平滑或LOS平滑
                    smoothing_method = self.config_.path_planning.post_smoothing_method
                    
                    if smoothing_method == 'catmull_rom':
                        # 使用Catmull-Rom曲线平滑
                        path = smooth_path(
                            path,
                            simplify_method=self.config_.path_planning.simplify_method,
                            simplify_threshold=self.config_.path_planning.simplify_threshold,
                            num_points_per_segment=self.config_.path_planning.num_points_per_segment,
                            curvature_threshold_deg=self.config_.path_planning.curvature_threshold_deg,
                            check_curvature=self.config_.path_planning.check_curvature
                        )
                    elif (
                        smoothing_method == 'los'
                        and self.config_.path_planning.enable_los_post_smoothing
                        and self.inflated_obstacle_ is not None
                    ):
                        # LOS平滑（向后兼容）
                        path = smooth_path_los(path, self.inflated_obstacle_)
            else:
                # 回退到传统A*
                logger.warning("代价图未初始化，使用传统A*算法")
                path = self.planner_.Plan(self.grid_, start, goal)
            
            if path:
                logger.info(f"路径规划成功，路径长度: {len(path)}")
            else:
                logger.warning("路径规划失败")
            
            return path
            
        except Exception as e:
            logger.error(f"路径规划失败: {e}")
            import traceback
            traceback.print_exc()
            return None

