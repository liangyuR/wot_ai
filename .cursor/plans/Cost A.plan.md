<!-- 60bcb8d4-17a5-483a-b4e5-82522cd671ba 68e5ef46-5c28-4039-80fa-16a0d2bf966b -->
# Cost A* 优化实施计划

## 阶段1：公共起点/终点修正逻辑实现

### 1.1 实现通用"最近可通行点"搜索helper

**文件**: `src/navigation/path_planner/astar_planner.py`

- 实现 `_find_nearest_free()` 函数：
- 输入：`grid: np.ndarray`, `start: Tuple[int, int]`, `max_radius: int`, `free_predicate: Callable[[float], bool]`
- 输出：找到的可通行点 `(x, y)` 或 `None`
- 使用BFS环形扩散，遍历4或8邻域
- 在半径超过 `max_radius` 后停止搜索

### 1.2 实现cost_map的start/goal修正包装

**文件**: `src/navigation/path_planner/astar_planner.py`

- 实现 `_adjust_start_goal_for_cost_astar()` 函数：
- 输入：`start, goal, cost_map, max_radius`
- 使用 `is_free_cost(v): return np.isfinite(v)` 作为判断条件
- 调用 `_find_nearest_free()` 修正start和goal
- 返回 `(new_start, new_goal)` 或 `None`

### 1.3 实现obstacle_map的start/goal修正包装

**文件**: `src/navigation/path_planner/astar_planner.py`

- 实现 `_adjust_start_goal_for_obstacle_astar()` 函数：
- 输入：`start, goal, inflated_obstacle_map, max_radius`
- 使用 `is_free_obstacle(v): return v == 0` 作为判断条件
- 调用 `_find_nearest_free()` 修正start和goal
- 返回 `(new_start, new_goal)` 或 `None`

## 阶段2：在cost_map A*中接入修正逻辑

### 2.1 在astar_with_cost入口处预处理start/goal

**文件**: `src/navigation/path_planner/astar_planner.py`

- 在 `astar_with_cost()` 函数开始处：
- 调用 `_adjust_start_goal_for_cost_astar()` 进行预处理
- 如果返回 `None`，记录warning并直接返回空列表
- 否则使用修正后的 `start, goal` 替代原值
- 添加日志：记录修正前后的坐标

### 2.2 配置max_radius参数

**文件**: `src/navigation/path_planner/astar_planner.py`

- 将 `max_radius` 设为常量或配置项：
- 建议值：10-15（针对256x256栅格）
- 可在函数参数中添加 `max_adjust_radius: int = 12`

## 阶段3：统一标准A*的逻辑

### 3.1 替换现有起点修正为公共helper

**文件**: `src/navigation/path_planner/astar_planner.py`

- 在 `AStar()` 方法中：
- 将现有的起点/终点修正逻辑（168-189行）替换为调用 `_adjust_start_goal_for_obstacle_astar()`
- 保持对外接口和日志行为不变
- 确保修正逻辑与cost_map A*一致

### 3.2 移除重复的_FindNearestWalkable方法

**文件**: `src/navigation/path_planner/astar_planner.py`

- 将 `_FindNearestWalkable()` 方法重构为通用的 `_find_nearest_free()`
- 或保留作为向后兼容的包装器，内部调用 `_find_nearest_free()`

## 阶段4：统计与日志增强

### 4.1 增加cost_map A*成功率统计

**文件**: `src/navigation/path_planner/path_planner_core.py`

- 在 `PathPlanningCore` 类中添加统计计数器：
- `_cost_astar_success_count: int = 0`
- `_cost_astar_fallback_count: int = 0`
- `_cost_astar_total_nodes: int = 0`
- 在 `plan()` 方法中：
- cost_map A*成功时增加成功计数和节点数统计
- cost_map A*失败回退时增加回退计数
- 定期打印统计信息（如每10次规划）

### 4.2 增强日志输出

**文件**: `src/navigation/path_planner/astar_planner.py`

- 在 `astar_with_cost()` 中：
- 当start被修正时：`cost_map A*: 起点 (sx, sy) 在障碍上，调整为 (sx', sy')`
- 当goal被修正时：`cost_map A*: 终点 (gx, gy) 在障碍上，调整为 (gx', gy')`
- 区分修正失败型和搜索失败型失败原因

**文件**: `src/navigation/path_planner/path_planner_core.py`

- 在回退日志中区分失败类型：
- 修正失败：`cost_map A* 规划失败（起点/终点修正失败），回退到传统 A*`
- 搜索失败：`cost_map A* 规划失败（搜索无解），回退到传统 A*`

## 阶段5：测试与验证

### 5.1 单元测试（可选）

**文件**: `utest/test_astar_adjustment.py`（新建）

- 测试 `_find_nearest_free()`：
- 起点在障碍内，附近有free cell
- 起点在边界，一侧是障碍
- 起点周围在max_radius内全是障碍
- 测试 `_adjust_start_goal_for_cost_astar()` 和 `_adjust_start_goal_for_obstacle_astar()`

### 5.2 集成测试场景

- 出生点在膨胀障碍内部：验证cost_map A*能通过修正成功规划
- 路径穿越狭窄通道：确保修正不会被吸到错误一侧
- 对比标准A*和cost_map A*的行为一致性

### To-dos

- [ ] 阶段1.1: 在MoveExecutor中添加平滑滤波（_smooth_forward, _smooth_turn, smoothing_alpha）
- [ ] 阶段1.2: 调整死区阈值（0.2→0.12）并应用于平滑后的值
- [ ] 阶段1.3: 实现前进方向滞回量化（_quantize_forward_with_hysteresis）
- [ ] 阶段1.4: 添加最小按键保持时间（可选，min_hold_time）
- [ ] 阶段1.5: 检查并统一stop_all()调用路径
- [ ] 阶段2.1: 优化MovementController角度减速策略（移除直接清零，添加min_forward_factor）
- [ ] 阶段2.2: 重构forward_cmd计算，降低角度误差影响，添加大角度衰减
- [ ] 阶段2.3: 扩展ControlConfig模型，添加所有MovementController参数
- [ ] 阶段2.3: 扩展ControlConfig，添加MoveExecutor参数（smoothing_alpha, deadzone等）
- [ ] 阶段2.3: 更新config.yaml，添加所有新控制参数
- [ ] 阶段2.3: 修改MovementService和NavigationRuntime，从配置读取参数
- [ ] 阶段3.1: 在PathFollowerWrapper中实现corridor概念（max_lateral_error, 横向偏差计算）
- [ ] 阶段3.2: 优化lookahead为基于距离的胡萝卜点（lookahead_distance）
- [ ] 阶段3.3: 放松waypoint切换条件（waypoint_switch_radius）
- [ ] 阶段3.4: 在ControlConfig和config.yaml中添加路径跟随参数
- [ ] 阶段4.1: 增强MoveExecutor和MovementController的调试日志输出
- [ ] 阶段1.1: 在MoveExecutor中添加平滑滤波（_smooth_forward, _smooth_turn, smoothing_alpha）
- [ ] 阶段1.2: 调整死区阈值（0.2→0.12）并应用于平滑后的值
- [ ] 阶段1.3: 实现前进方向滞回量化（_quantize_forward_with_hysteresis）
- [ ] 阶段1.4: 添加最小按键保持时间（可选，min_hold_time）
- [ ] 阶段1.5: 检查并统一stop_all()调用路径
- [ ] 阶段2.1: 优化MovementController角度减速策略（移除直接清零，添加min_forward_factor）
- [ ] 阶段2.2: 重构forward_cmd计算，降低角度误差影响，添加大角度衰减
- [ ] 阶段2.3: 扩展ControlConfig模型，添加所有MovementController参数
- [ ] 阶段2.3: 扩展ControlConfig，添加MoveExecutor参数（smoothing_alpha, deadzone等）
- [ ] 阶段2.3: 更新config.yaml，添加所有新控制参数
- [ ] 阶段2.3: 修改MovementService和NavigationRuntime，从配置读取参数
- [ ] 阶段3.1: 在PathFollowerWrapper中实现corridor概念（max_lateral_error, 横向偏差计算）
- [ ] 阶段3.2: 优化lookahead为基于距离的胡萝卜点（lookahead_distance）
- [ ] 阶段3.3: 放松waypoint切换条件（waypoint_switch_radius）