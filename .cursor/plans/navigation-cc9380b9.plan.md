<!-- cc9380b9-c3bc-41e4-b3bd-eb55b8408680 24cb9e8f-592b-4242-96f8-e4c76ac0867c -->
# 导航控制重构计划

## 目标

- 合并 `NavigationMain` 与 `AIController` 为单一导航控制器，统一管理初始化、运行、停止。
- BattleTask 在检测到地图后触发掩码更新，并控制导航 AI 的启动/停止。
- YOLO 模型在程序启动时预热，战斗期间可动态启停检测与路径规划。

## 步骤

1. **整理现状**  

- `src/navigation/navigation_main.py`：了解现有服务初始化、掩码加载、运行逻辑。  
- `src/core/ai_controller.py`：梳理生命周期管理、线程、掩码更新流程。  
- `src/core/battle_task.py`：确认战斗状态流转、地图识别、AI控制调用点。

2. **设计新控制器结构**  

- 拟定统一类（例如 `NavigationController`）：负责 YOLO 预热（程序启动时调用）、战斗阶段的掩码加载、检测线程启动/停止、路径规划控制。  
- 明确 API：`initialize()`、`prepare_battle(map_name)`、`start()`、`stop()`、`update_mask(map_name)` 等，供 BattleTask 调用。  
- 设计内部模块接口（捕获、检测、规划、控制）及事件/回调，以便独立开发与测试。

3. **实现控制器合并并解耦组件**  

- 在 `navigation/navigation_main.py` 或新文件中实现统一的 `NavigationController`，迁移 `NavigationMain`/`AIController` 的必要逻辑。  
- 在 `initialize()` 中加载并预热 YOLO，同时提前实例化控制服务、屏幕捕获服务、路径规划服务，使其处于“待命”状态，实际运行由 BattleTask 触发。  
- 通过接口/回调让捕获、检测、规划、控制等模块以最小依赖协作，例如为检测结果提供观察者或队列。  
- `prepare_battle(map_name)` 根据 BattleTask 提供的地图名称动态加载掩码，不再依赖配置中的固定路径；加载完成后将数据交由规划组件。  
- `start()`/`stop()` 负责线程调度，确保检测与控制可多次启停。

4. **调整 BattleTask 流程**  

- `_handle_garage_state`：如导航控制器未初始化则调用 `initialize()` 预热；之后只负责车辆选择与进入战斗。  
- `_handle_battle_state`：地图识别后调用 `prepare_battle(map_name)`，成功后再调用 `start()` 启动检测/导航；状态结束或返回车库时调用 `stop()`。  
- 移除旧 `AIController` 依赖，更新属性与调用点。

5. **配置与模板更新**  

- 清理 `config/config.yaml.template` 中与固定掩码路径相关的字段，仅保留可选目录或相对路径设置。  
- `global_path` 等工具函数提供模板/掩码基础路径，由控制器在运行时决定最终文件。

### To-dos

- [ ] 梳理现有导航/AI控制器代码结构
- [ ] 设计统一 NavigationController API
- [ ] 实现并合并导航控制器逻辑
- [ ] 接入新控制器并调整流程
- [ ] 调整配置模板与路径解析验证整体验证流程