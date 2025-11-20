#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional
from loguru import logger


class PathPlannerWrapper:
    """
    规划包装器（对 PathPlanningService 的封装）

    负责：
    - 调用 A* / Dijkstra / 任意 planner 产生 grid_path
    - 转换成 minimap 坐标 path_world（用于路径跟随）
    """

    def __init__(self, planner, minimap_size: Tuple[int, int], scale_xy: Tuple[float, float]):
        """
        Args:
            planner: PathPlanningService 实例
            minimap_size: (minimap_width, minimap_height)
            scale_xy: (sx, sy) 每个 grid 单位到 minimap 像素的缩放
        """
        self.planner = planner
        self.minimap_w, self.minimap_h = minimap_size
        self.sx, self.sy = scale_xy

    def plan(self, detections) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
        """
        返回 (grid_path, world_path)
        grid_path: [(gx, gy), ...]   - 网格路径
        world_path: [(wx, wy), ...]  - minimap 坐标路径

        若规划失败：返回 ([], [])
        """

        try:
            # 输入是 minimap 大小 + detections
            grid_path = self.planner.plan_path(
                (self.minimap_w, self.minimap_h),
                detections
            )
        except Exception as e:
            logger.error(f"路径规划异常: {e}")
            return [], []

        if not grid_path:
            logger.warning("路径规划失败或结果为空")
            return [], []

        # 转换为 minimap 坐标 path_world
        path_world = [(gx * self.sx, gy * self.sy) for gx, gy in grid_path]

        return grid_path, path_world
