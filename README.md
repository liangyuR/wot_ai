# WoT AI

《坦克世界》自动化与导航助手，包含状态机战斗流程、视觉检测、小地图路径规划、键盘鼠标控制，以及调试视图。

## 运行环境
- Windows 10/11
- Python 3.11
- Poetry 依赖管理
- CUDA

## 安装与虚拟环境
```bash
pip install poetry
```

Poetry 2.0 之后推荐使用 `env activate`：
```bash
poetry env activate
```
该命令会输出一条激活命令（如 `...\\Scripts\\activate.ps1`），复制并执行即可进入虚拟环境。

激活后安装依赖
poetry install

## CUDA 版本查看
在命令行执行以下任一命令：
```bash
nvidia-smi
```
提示：`nvidia-smi` 显示驱动支持的 CUDA 版本；

## PyTorch 安装
建议根据 CUDA 版本选择官方安装命令（Windows + pip 示例）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
请根据你机器上的 CUDA 版本替换 `cu121`，以 PyTorch 官方给出的命令为准。

## 配置
配置文件在 `config/config.yaml`，按需修改：
- `game.exe_path` : 游戏路径

路径可以使用相对路径（相对程序目录）或绝对路径。

## 运行方式
主界面：
```bash
# 1. 激活虚拟环境，参考上面
# 2. 在项目根目录中执行
python ./src/main_window.py
```

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

## 模型
可通过邮件联系：1160158693@qq.com
