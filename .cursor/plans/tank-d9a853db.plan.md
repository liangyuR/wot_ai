<!-- d9a853db-8e92-4952-bb83-b217ce1f7911 939dc145-590e-4200-a217-72f38f7abe69 -->
# 坦克选择流程重构计划

## 目标

- 遵循“策略 (TankSelector) → 指令 (BattleTask) → UI执行 (UIFlow/UIActions)”分层。
- 去除三层重复的模板匹配/点击代码，统一由 UI 层执行。

## 步骤

1. **现状梳理**

- 记录 `TankSelector.pick()` 输出内容、`BattleTask.select_tank()` 的职责与依赖、`UIFlow._SelectTank()` 的现状。
- 明确目前谁持有 `TankSelector`、`UIActions` 等实例。

2. **接口设计**

- 让 `TankSelector` 专注返回待选车辆模板（文件名/路径/附带 metadata）。
- 扩展 `UIFlow/UIActions`，提供 `SelectVehicle(template_name, template_dir)` 或类似接口，内部只用 `ClickTemplate`。

3. **调用链改造**

- 在 `BattleTask.select_tank()` 中，注入 `TankSelector` 和 UI 控制对象，改为“先从 `TankSelector` 获取模板，再调用 UI 层点击”，去掉直接 `match_template`。
- 如果 `UIFlow._SelectTank()` 仍存在，则其内部不再自己匹配，而是调用新的 UI 接口；或直接由 `BattleTask` 调用 UI 层。

4. **配置整理**

- 统一 `vehicle_screenshots` 路径配置入口，置信度等参数不再散落在各文件。
- 更新 `TaskManager/MainWindow` 的依赖注入，确保 `BattleTask` 能拿到 `TankSelector` 和 UI 控制实例。

5. **回归验证**

- 检查 `tank_selector.py`、`battle_task.py`、`ui_flow.py`、`main_window.py` 等是否遵循新分层。
- 审阅日志/异常路径，确保易于调试。

## 实施 TODOs

- `audit-current`: 梳理坦克选择调用链与依赖
- `design-api`: 定义 TankSelector 输出与 UI 执行接口契约
- `update-callers`: 改造 BattleTask/UIFlow 调用链
- `config-cleanup`: 统一模板目录与置信度配置
- `verify-flow`: 回归检查调用路径与日志

### To-dos

- [ ] 梳理坦克选择调用链与参数
- [ ] 实现 TankSelector 统一点击接口
- [ ] BattleTask/UIFlow 改用统一接口
- [ ] 整理模板目录/置信度配置
- [ ] 回归检查入口与调用流程