# src/navigation/core/path_planner_core.py
from loguru import logger

from src.navigation.path_planner.map_model import MapModel, PlanRequest, PlanResult
from src.navigation.path_planner.astar_planner import AStarPlanner, astar_with_cost
from src.navigation.path_planner.path_smoothing import smooth_path_los, smooth_path

class PathPlanningCore:
    """纯路径规划器：只关心栅格/代价图，不关心 minimap / det。"""

    def __init__(
        self,
        enable_astar_smoothing: bool,
        astar_smooth_weight: float,
        post_smoothing_method: str,
        simplify_method: str,
        simplify_threshold: float,
        num_points_per_segment: int,
        curvature_threshold_deg: float,
        check_curvature: bool,
        enable_los_post_smoothing: bool,
    ) -> None:
        self._planner = AStarPlanner(
            enable_smoothing=enable_astar_smoothing,
            smooth_weight=astar_smooth_weight,
        )
        self._post_method = post_smoothing_method
        self._simplify_method = simplify_method
        self._simplify_threshold = simplify_threshold
        self._num_points_per_segment = num_points_per_segment
        self._curv_th_deg = curvature_threshold_deg
        self._check_curv = check_curvature
        self._enable_los = enable_los_post_smoothing
        
        # cost_map A* 统计计数器
        self._cost_astar_success_count: int = 0
        self._cost_astar_fallback_count: int = 0
        self._cost_astar_adjust_fail_count: int = 0  # 修正失败计数
        self._cost_astar_search_fail_count: int = 0  # 搜索失败计数
        self._cost_astar_total_nodes: int = 0
        self._plan_count: int = 0

    def plan(self, map_model: MapModel, req: PlanRequest) -> PlanResult:
        """在给定地图上做一次路径规划（栅格坐标）。"""

        grid = map_model.grid
        cost_map = map_model.cost_map
        inflated = map_model.inflated_obstacle
        start, goal = req.start, req.goal

        try:
            # 1) 优先尝试 cost_map A*
            path = None
            fallback_reason = None
            if cost_map is not None:
                path = astar_with_cost(cost_map, start, goal)
                if not path:
                    # 检查失败原因（通过日志判断，或修改astar_with_cost返回更多信息）
                    # 这里我们通过检查astar_with_cost的日志来判断
                    # 实际上，astar_with_cost在修正失败时会记录特定日志
                    # 为了更准确，我们可以通过检查是否在障碍上来判断
                    import numpy as np
                    h, w = cost_map.shape
                    sx, sy = start
                    gx, gy = goal
                    
                    # 检查是否是因为修正失败（起点/终点在障碍上且无法修正）
                    start_blocked = (0 <= sx < w and 0 <= sy < h and not np.isfinite(cost_map[sy, sx]))
                    goal_blocked = (0 <= gx < w and 0 <= gy < h and not np.isfinite(cost_map[gy, gx]))
                    
                    if start_blocked or goal_blocked:
                        # 尝试修正，如果修正失败则认为是修正失败
                        from src.navigation.path_planner.astar_planner import _adjust_start_goal_for_cost_astar
                        adjusted = _adjust_start_goal_for_cost_astar(start, goal, cost_map, max_radius=100)
                        if adjusted is None:
                            self._cost_astar_adjust_fail_count += 1
                            fallback_reason = "起点/终点修正失败"
                            logger.warning(f"cost_map A* 规划失败（{fallback_reason}），回退到传统 A*")
                        else:
                            self._cost_astar_search_fail_count += 1
                            fallback_reason = "搜索无解"
                            logger.warning(f"cost_map A* 规划失败（{fallback_reason}），回退到传统 A*")
                    else:
                        self._cost_astar_search_fail_count += 1
                        fallback_reason = "搜索无解"
                        logger.warning(f"cost_map A* 规划失败（{fallback_reason}），回退到传统 A*")
                    
                    self._cost_astar_fallback_count += 1
                    path = None
                else:
                    # cost_map A* 成功
                    self._cost_astar_success_count += 1
                    # 注意：astar_with_cost内部已经记录了节点数，这里我们无法直接获取
                    # 如果需要精确统计，需要修改astar_with_cost的返回值

            # 2) Fallback 到传统 A*
            if not path:
                path = self._planner.Plan(grid, start, goal)

            if not path:
                return PlanResult(ok=False, path=[], reason="A* 规划失败")

            # 更新规划计数并定期打印统计
            self._plan_count += 1
            if self._plan_count % 10 == 0:
                total_attempts = self._cost_astar_success_count + self._cost_astar_fallback_count
                if total_attempts > 0:
                    success_rate = self._cost_astar_success_count / total_attempts * 100
                    logger.info(
                        f"cost_map A* 统计 (最近{total_attempts}次尝试): "
                        f"成功率={success_rate:.1f}% ({self._cost_astar_success_count}/{total_attempts}), "
                        f"修正失败={self._cost_astar_adjust_fail_count}, "
                        f"搜索失败={self._cost_astar_search_fail_count}"
                    )

            # 3) 后处理平滑
            if self._post_method == "catmull_rom":
                path = smooth_path(
                    path,
                    simplify_method=self._simplify_method,
                    simplify_threshold=self._simplify_threshold,
                    num_points_per_segment=self._num_points_per_segment,
                    curvature_threshold_deg=self._curv_th_deg,
                    check_curvature=self._check_curv,
                )
            elif (
                self._post_method == "los"
                and self._enable_los
                and inflated is not None
            ):
                path = smooth_path_los(path, inflated)

            return PlanResult(ok=True, path=path or [], reason="ok")

        except Exception as e:
            logger.error(f"CorePathPlanner 规划异常: {e}")
            import traceback; traceback.print_exc()
            return PlanResult(ok=False, path=[], reason=str(e))


if __name__ == "__main__":
    # 简单自测：在 10x10 网格上绕开障碍走一条路
    import numpy as np

    # 1. 构造一个简单的障碍地图：中间一列是墙，中间开一个门
    h, w = 10, 10
    grid = np.zeros((h, w), dtype=np.uint8)  # 0 = free
    grid[:, 5] = 1                           # 整列障碍
    grid[5, 5] = 0                           # 中间开个口

    # 2. 把 0/1 grid 变成 cost_map：free = 1.0, 障碍 = inf
    cost_map = np.full((h, w), 1.0, dtype=np.float32)
    cost_map[grid == 1] = np.inf

    # 3. 创建核心规划器
    core = PathPlanningCore(enable_astar_smoothing=True, astar_smooth_weight=0.3, post_smoothing_method="catmull_rom", simplify_method="rdp", simplify_threshold=8.0, num_points_per_segment=8, curvature_threshold_deg=40.0, check_curvature=True, enable_los_post_smoothing=True)

    start = (0, 0)   
    goal  = (9, 9)   

    map_model = MapModel(grid=grid, cost_map=cost_map, inflated_obstacle=None, size=(w, h))
    plan_request = PlanRequest(start=start, goal=goal)
    plan_result = core.plan(map_model, plan_request)

    print("规划结果 PlanResult:")
    print(f"ok: {plan_result.ok}")
    print(f"path: {plan_result.path}")
    print(f"reason: {plan_result.reason}")

    # 4. 用 ASCII 粗暴可视化： 
    #   '#' = 障碍, '.' = 空地, '*' = 路径, 'S' = 起点, 'G' = 终点
    vis = np.full((h, w), '.', dtype=str)

    for y in range(h):
        for x in range(w):
            if np.isinf(cost_map[y, x]):
                vis[y, x] = '#'

    if plan_result and getattr(plan_result, "ok", False) and plan_result.path:
        for coord in plan_result.path:
            x, y = int(coord[0]), int(coord[1])
            vis[y, x] = '*'

        # 标记起点终点
        sx, sy = plan_request.start
        gx, gy = plan_request.goal
        vis[sy, sx] = 'S'
        vis[gy, gx] = 'G'

    print("\nASCII 地图：")
    for row in vis:
        print("".join(row))
