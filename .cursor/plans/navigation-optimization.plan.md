# 导航代码优化计划

## 目标

简化、优化导航部分代码，提升性能和可维护性。

## 主要任务

### 1. 删除调试用的代码

- 查找并删除所有 `logger.debug()` 调用（保留必要的 info/warning/error）
- 删除所有 `print()` 语句
- 删除调试相关的临时变量和注释
- 删除 FPS 显示相关的调试代码（如果不需要）
- 清理 `transparent_overlay.py` 中的调试日志

### 2. 删除目前结构不需要的代码

- 检查未使用的导入
- 删除未使用的类、函数、方法
- 检查是否有重复的实现
- 删除过时的接口和兼容代码
- 简化 `overlay_dpg.py` 中未使用的方法（如 `DrawPath` 如果未被调用）
- 检查 `common/` 目录下是否有未使用的配置

### 3. 性能优化

- 优化路径缓存机制（减少重复计算）
- 优化掩码纹理更新（只在变化时更新）
- 减少不必要的数组拷贝
- 优化队列大小和阻塞策略
- 优化线程同步机制（减少锁竞争）
- 优化坐标转换计算（缓存结果）
- 减少 logger 调用频率（批量日志）

### 4. UI导航框优化（关键）

**当前问题：**

- Overlay 显示在左上角，不在小地图上
- 可能被 mss 捕获

**目标：**

- 完全透明，直接绘制到真实的小地图上
- 不能被 mss 图像捕获

**实现方案：**

- **方案A：使用 Windows GDI 直接绘制到屏幕DC**
- 使用 `GetDC(NULL)` 获取屏幕设备上下文
- 使用 GDI+ 或 GDI 直接绘制路径、点等
- 设置绘制区域为小地图位置
- 这种方式不会被 mss 捕获（因为直接绘制到屏幕）
- 需要处理窗口刷新（游戏窗口移动/最小化时）

- **方案B：使用 DirectX/OpenGL Hook**
- Hook 游戏的渲染API
- 在游戏渲染时叠加绘制
- 完全透明，不会被捕获
- 实现复杂，需要了解游戏使用的渲染API

- **方案C：使用 Windows Layered Window + 特殊属性**
- 创建完全透明的分层窗口
- 使用 `WS_EX_LAYERED` + `WS_EX_TRANSPARENT`
- 设置窗口位置为小地图位置
- 使用 `SetLayeredWindowAttributes` 设置完全透明（alpha=0）
- 但绘制内容使用不透明颜色
- 这种方式可能仍会被 mss 捕获

**推荐方案：方案A（Windows GDI 直接绘制）**

- 实现相对简单
- 不会被 mss 捕获
- 性能较好
- 需要处理窗口刷新和重绘

**实现步骤：**

1. 创建新的 `DirectDrawOverlay` 类替代 `TransparentOverlay`
2. 使用 Windows GDI/GDI+ API 直接绘制到屏幕
3. 在独立线程中定期重绘（处理窗口刷新）
4. 绘制路径、起点、终点、角度箭头等元素
5. 确保绘制区域精确对应小地图位置
6. 处理游戏窗口移动/最小化/恢复等事件

## 文件修改清单

### 需要修改的文件

- `wot_ai/game_modules/navigation/ui/transparent_overlay.py` - 重构为直接绘制实现
- `wot_ai/game_modules/navigation/ui/overlay_dpg.py` - 可能删除或简化
- `wot_ai/game_modules/navigation/service/minimap_service.py` - 更新 overlay 初始化逻辑
- `wot_ai/game_modules/navigation/service/thread_manager.py` - 优化性能，删除调试代码
- `wot_ai/game_modules/navigation/navigation_main.py` - 删除调试代码
- `wot_ai/game_modules/navigation/core/` - 检查并删除未使用的模块

### 可能需要删除的文件

- 如果 `overlay_dpg.py` 完全被替代，可以删除
- 检查 `common/overlay_config.py` 是否还需要

## 注意事项

- 保持向后兼容性（如果可能）
- 确保性能优化不影响功能
- UI 优化需要测试各种游戏窗口状态
- 确保绘制不会影响游戏性能
- 处理多显示器情况