# WoT AI

《坦克世界》自动化与导航助手，包含状态机战斗流程、视觉检测、小地图路径规划、键盘鼠标控制，以及调试视图。

## 运行环境
- Windows 10/11
- Python 3.11
- Poetry 依赖管理
- 建议安装可用的 GPU 驱动（若使用 CUDA 模型加速）

## 安装
```bash
pip install poetry
poetry install
```

## Poetry 虚拟环境
Poetry 2.0 之后推荐使用 `env activate`：  
```bash
poetry env activate
```
根据提示将其输出的命令粘贴执行（会进入当前项目虚拟环境）。

不进入 shell 的临时执行方式：
```bash
poetry run python src/main_window.py
```

如果需要恢复 `poetry shell`，可安装 shell 插件（参考官方文档）。

## CUDA 版本查看
在命令行执行以下任一命令：
```bash
nvidia-smi
```
或：
```bash
nvcc --version
```
提示：`nvidia-smi` 显示驱动支持的 CUDA 版本；`nvcc --version` 显示本机安装的 CUDA Toolkit 版本。

## PyTorch 安装
建议根据 CUDA 版本选择官方安装命令（Windows + pip 示例）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
如果使用 CPU 版本：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
请根据你机器上的 CUDA 版本替换 `cu121`，以 PyTorch 官方给出的命令为准。

## 配置
配置文件在 `config/config.yaml`，按需修改：
- `model.base_path` / `model.arrow_path`: YOLO 模型路径
- `game.exe_path` / `game.process_name`: 游戏路径与进程名
- `minimap.size` / `grid.size`: 小地图与栅格参数
- `vehicle.screenshot_dir`: 车辆截图目录（默认 `resource/vehicle_screenshots`）
- `ui.enable` / `ui.overlay_fps`: 调试视图与刷新帧率
- `failure_dump.*`: 失败样本保存路径与频率

路径可以使用相对路径（相对程序目录）或绝对路径。

## 运行方式
主界面：
```bash
poetry run python src/main_window.py
```

导航独立模式（热键控制）：
```bash
poetry run python src/navigation/nav_runtime/navigation_runtime.py
```
热键说明：
- F9: 初始化并开始导航（自动检测地图）
- F10: 结束本局导航
- F11: 暂停/恢复导航
- ESC: 退出并释放资源

## 使用流程
1) 准备模型与模板资源，确认 `config/config.yaml` 中路径正确。
2) 车辆截图放入 `resource/vehicle_screenshots/`，按优先级命名为 `1.png`、`2.png`、`3.png` 等。
3) 启动游戏后运行主界面，点击“开始导航”进入自动流程。
4) 运行日志位于 `Logs/`，失败样本按 `failure_dump` 配置保存。

## 项目结构
- `src/main_window.py`: 主 UI 入口
- `src/core/`: 状态机与战斗任务
- `src/navigation/`: 导航运行时、路径规划、控制与采集服务
- `src/vision/`: 视觉检测与地图识别
- `src/gui/`: 调试视图
- `src/utils/`: 通用工具与路径配置
- `config/`: 运行时配置
- `resource/`: 模板/模型/静态资源
- `utest/`: 测试用例与夹具
- `Logs/`: 运行日志

## 测试
```bash
poetry run pytest utest -q
```

## 打包（可选）
```bash
poetry run pyinstaller wot-robot.spec
```

## 常见问题
- 模型/模板找不到：检查 `config/config.yaml` 路径，必要时改为绝对路径。
- 无法控制或捕获窗口：确认游戏与程序权限一致，且窗口未被最小化。
- 性能不足：适当降低 `detection.max_fps` 或提高置信度阈值。

# 模型
可通过邮件联系：1160158693@qq.com