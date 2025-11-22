#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PathPlanningService

中间层：
- 负责加载掩码 → 构建 MapModel
- 做 world<->grid 坐标转换
- 调用底层 PathPlanner 进行栅格路径规划
- 输出：grid_path + world_path，供 NavigationRuntime 使用
"""

from typing import Optional, List, Tuple
from loguru import logger

import numpy as np

from src.navigation.config.loader import load_config
from src.navigation.config.models import NavigationConfig
from src.vision.minimap_detector import MinimapDetectionResult
from src.navigation.core.coordinate_utils import world_to_grid, grid_to_world
from src.utils.mask_loader import load_mask
from src.utils.global_path import GetMapMaskPath

from src.navigation.path_planner.map_model import (
    MapModel,
    PlanRequest,
    PlanResult,
)
from src.navigation.path_planner.path_planner_core import PathPlanningCore


class PathPlanningService:
    """
    路径规划服务（中间层）

    生命周期大致是：

    1. 创建实例：pps = PathPlanningService(cfg)
    2. 当前地图加载完成后调用：pps.load_map(map_name, minimap_size)
    3. 控制线程每 tick 调用：pps.plan_path(detections, minimap_size)
       拿到 grid_path / world_path
    """

    def __init__(self, cfg: NavigationConfig) -> None:
        self.cfg = cfg

        # 当前地图的 minimap 像素尺寸（width, height）
        self.minimap_size: Optional[Tuple[int, int]] = None

        # MapModel + 底层规划器
        self._map_model: Optional[MapModel] = None
        self._planner: Optional[PathPlanningCore] = None

    # ------------------------------------------------------------------
    # 地图 / 掩码加载
    # ------------------------------------------------------------------
    def load_map(self, map_name: str, minimap_size: Tuple[int, int]) -> bool:
        """
        加载当前地图的掩码，并构建 MapModel + PathPlanner。

        Args:
            map_name: 地图名称（用于拼 mask 路径）
            minimap_size: 小地图像素尺寸 (width, height)

        Returns:
            bool: 是否加载成功
        """
        self.minimap_size = minimap_size
        grid_w, grid_h = self.cfg.grid.size
        grid_size = (grid_w, grid_h)

        mask_path = GetMapMaskPath(map_name)
        logger.info(
            f"[PathPlanningService] 加载地图掩码: map={map_name}, mask_path={mask_path}, "
            f"minimap_size={minimap_size}, grid_size={grid_size}"
        )

        try:
            # 1) 调用通用 mask_loader
            md = load_mask(
                mask_path=str(mask_path),
                minimap_size=minimap_size,
                grid_size=grid_size,
                inflation_radius_px=self.cfg.mask.inflation_radius_px,
                cost_alpha=self.cfg.mask.cost_alpha,
            )

            # 2) 构建 MapModel
            #   注意：这里假设 MapModel 有 inflated_obstacle 字段；
            #   如果你当前版本没有，删掉对应字段即可。
            self._map_model = MapModel(
                grid=md.grid,
                cost_map=md.cost_map,
                inflated_obstacle=md.inflated_obstacle,
                size=grid_size,
            )

            # 3) 初始化底层 PathPlanner
            self._planner = PathPlanningCore(
                enable_astar_smoothing=self.cfg.path_planning.enable_astar_smoothing,
                astar_smooth_weight=self.cfg.path_planning.astar_smooth_weight,
                post_smoothing_method=self.cfg.path_planning.post_smoothing_method,
                simplify_method=self.cfg.path_planning.simplify_method,
                simplify_threshold=self.cfg.path_planning.simplify_threshold,
                num_points_per_segment=self.cfg.path_planning.num_points_per_segment,
                curvature_threshold_deg=self.cfg.path_planning.curvature_threshold_deg,
                check_curvature=self.cfg.path_planning.check_curvature,
                enable_los_post_smoothing=self.cfg.path_planning.enable_los_post_smoothing,
            )

            logger.info(
                f"[PathPlanningService] 地图加载成功: grid_shape={md.grid.shape}, "
                f"cost_map_shape={md.cost_map.shape}"
            )
            return True

        except Exception as e:
            logger.error(f"[PathPlanningService] 加载地图/掩码失败: {e}")
            import traceback
            traceback.print_exc()
            self._map_model = None
            self._planner = None
            return False

    # ------------------------------------------------------------------
    # 路径规划主接口
    # ------------------------------------------------------------------
    def plan_path(
        self,
        detections: MinimapDetectionResult,
    ) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[float, float]]]]:
        """
        对当前检测结果进行路径规划。

        输入：
            - detections.self_pos: 当前己方坐标（minimap 像素）
            - detections.enemy_flag_pos (或其它目标点): 目标坐标（minimap 像素）

        输出：
            - grid_path: 栅格路径（grid 坐标系）
            - world_path: 像素路径（minimap 坐标系）
        """

        # 基础检查
        if self._planner is None or self._map_model is None or self.minimap_size is None:
            logger.error("[PathPlanningService] map 未初始化，请先调用 load_map()")
            return None, None

        if detections.self_pos is None or detections.enemy_flag_pos is None:
            logger.warning("[PathPlanningService] 检测结果不完整（缺少 self_pos / enemy_flag_pos），无法规划路径")
            return None, None

        try:
            minimap_size = self.minimap_size
            grid_size = self._map_model.size

            # ----------------------------------------------------------
            # 1) world(minimap 像素) → grid 栅格坐标
            # ----------------------------------------------------------
            start_grid = world_to_grid(
                detections.self_pos,
                minimap_size=minimap_size,
                grid_size=grid_size,
            )
            goal_grid = world_to_grid(
                detections.enemy_flag_pos,
                minimap_size=minimap_size,
                grid_size=grid_size,
            )

            logger.info(
                f"[PathPlanningService] 规划请求: "
                f"start_world={detections.self_pos}, goal_world={detections.enemy_flag_pos}, "
                f"start_grid={start_grid}, goal_grid={goal_grid}"
            )

            # 越界简单保护
            gx, gy = start_grid
            if not (0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]):
                logger.warning(f"[PathPlanningService] 起点超出 grid 范围: {start_grid}")
                return None, None

            gx, gy = goal_grid
            if not (0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]):
                logger.warning(f"[PathPlanningService] 终点超出 grid 范围: {goal_grid}")
                return None, None

            # ----------------------------------------------------------
            # 2) 构造 PlanRequest，调用底层 PathPlanner
            # ----------------------------------------------------------
            req = PlanRequest(start=start_grid, goal=goal_grid)
            result: PlanResult = self._planner.plan(self._map_model, req)

            if not result.ok or not result.path:
                logger.warning(f"[PathPlanningService] 路径规划失败: {result.reason}")
                return None, None

            grid_path = result.path
            logger.info(f"[PathPlanningService] 路径规划成功，节点数: {len(grid_path)}")

            # ----------------------------------------------------------
            # 3) grid → world(minimap 像素) 路径
            # ----------------------------------------------------------
            world_path: List[Tuple[float, float]] = [
                grid_to_world(p, minimap_size=minimap_size, grid_size=grid_size)
                for p in grid_path
            ]

            return grid_path, world_path

        except Exception as e:
            logger.error(f"[PathPlanningService] plan_path 异常: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def get_grid_mask(self) -> np.ndarray:
        return self._map_model.grid

if __name__ == "__main__":
    import cv2
    import numpy as np
    from loguru import logger
    from src.utils.global_path import GetConfigPath
    from src.utils.mask_loader import _imread_gray
    import argparse

    # 1. 配置 & 路径规划服务
    cfg = load_config(str(GetConfigPath()))
    logger.info(f"cfg: {cfg.path_planning}, mask: {cfg.mask}")

    parser = argparse.ArgumentParser()
    parser.add_argument("map_name", type=str, help="地图名称")
    args = parser.parse_args()
    map_name = args.map_name

    # 用真实掩码的尺寸作为 minimap_size
    mask_path = GetMapMaskPath(map_name) 
    img = _imread_gray(mask_path)
    if img is None:
        raise RuntimeError(f"读取掩码失败: {mask_path}")
    h, w = img.shape[:2]
    minimap_size = (w, h)

    planner = PathPlanningService(cfg)
    if not planner.load_map(map_name, minimap_size):
        raise RuntimeError("load_map 失败")

    # 2. 在掩码中选两个可通行点（白色=255）
    ys, xs = np.where(img == 255)
    if xs.size < 2:
        raise RuntimeError("掩码里可通行像素太少，没法选起点终点")

    # 靠左的一个当 start，靠右的一个当 goal
    h0, w0 = img.shape[:2]
    scale_x = minimap_size[0] / w0
    scale_y = minimap_size[1] / h0

    ys, xs = np.where(img == 255)

    start_world = (
        int(xs.min()  * scale_x),
        int(ys[np.argmin(xs)] * scale_y),
    )
    goal_world  = (
        int(xs.max()  * scale_x),
        int(ys[np.argmax(xs)] * scale_y),
    )

    logger.info(f"demo 起点(像素): {start_world}, 终点(像素): {goal_world}")

    # 3. 伪造一次检测结果（只用 self_pos / enemy_flag_pos）
    det = MinimapDetectionResult(
        self_pos=start_world,
        enemy_flag_pos=goal_world,
        self_angle=0.0,           # demo 用不到，可以随便填
        raw_detections=[],
    )

    grid_path, world_path = planner.plan_path(det)
    if not world_path:
        logger.error("规划失败")
        exit(1)

    logger.info(f"规划成功, grid_path_len={len(grid_path)}, world_path_len={len(world_path)}")

    # 4. 可视化：把路径画到掩码上，方便肉眼检查
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 起点终点
    cv2.circle(vis, start_world, 5, (0, 255, 0), -1)   # 绿 = 起点
    cv2.circle(vis, goal_world, 5, (0, 0, 255), -1)    # 红 = 终点

    # 路径
    for i in range(1, len(world_path)):
        x1, y1 = map(int, world_path[i - 1])
        x2, y2 = map(int, world_path[i])
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色路径

    out_path = "path_planning_demo.png"
    cv2.imwrite(out_path, vis)
    logger.info(f"demo 可视化已保存到: {out_path}")