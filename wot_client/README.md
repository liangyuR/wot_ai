# WoT Game Client

游戏客户端，运行在 Windows 环境。

## ✨ 功能

### 屏幕捕获
- **🎯 智能窗口捕获**（推荐）- 通过进程名自动定位游戏窗口
- **🖥️ 全屏捕获** - 捕获整个屏幕
- **⚡ C++ 加速** - 高性能屏幕捕获（可选）
- **🔄 动态分辨率** - 自适应窗口大小

### 操作控制
- 键盘鼠标模拟
- 接收并执行 AI 操作指令
- 与训练服务器通信

### 数据采集
- 录制游戏画面和操作
- 保存为 JPEG 帧或视频（可选）
- 实时保存，避免内存溢出
- 支持帧采样，节省存储
- 用于模仿学习训练

## 🚀 快速开始

### 1. 安装

```bash
cd wot_client

# 方式 1: 使用批处理脚本（推荐）
start_recording.bat

# 方式 2: 手动安装
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 系统诊断（推荐）

```bash
# 检查系统状态
python test_capture.py

# 测试窗口捕获
python test_window_capture.py
```

### 3. 开始录制

```bash
# 方式 1: 默认配置（推荐）- 只保存帧，全屏模式
python record_main.py

# 方式 2: 每 5 帧保存一次（节省存储）
python record_main.py --frame-step 5

# 方式 3: 窗口模式 - 自动定位游戏
python record_main.py --mode window --process WorldOfTanks.exe

# 方式 4: 同时保存帧和视频
python record_main.py --save-format both

# 或使用批处理脚本
start_recording.bat
```

## 📚 文档

- [**性能优化指南**](PERFORMANCE_GUIDE.md) - 解决卡顿问题（⭐新用户必读）
- [**窗口捕获指南**](WINDOW_CAPTURE_GUIDE.md) - 智能窗口捕获完整教程
- [**数据格式说明**](DATA_FORMAT.md) - Session 数据结构详解
- [**快速开始**](QUICKSTART_RECORDING.md) - 5 分钟入门指南
- [**故障排查**](TROUBLESHOOTING.md) - 详细的问题解决方案
- [**更新日志**](CHANGELOG.md) - 功能更新和改进记录

## 🎯 两种捕获模式

### Window 模式（推荐）✅

```bash
python record_main.py --mode window --process WorldOfTanks.exe
```

**优点**：
- ✅ 自动定位游戏窗口
- ✅ 窗口移动时自动跟随
- ✅ 只捕获游戏内容
- ✅ 支持多显示器

**要求**：游戏使用窗口化或窗口化全屏模式

### Fullscreen 模式

```bash
python record_main.py --mode fullscreen
```

**优点**：
- ✅ 支持完全全屏模式
- ✅ 无需额外依赖

**缺点**：
- ❌ 捕获整个屏幕
- ❌ 需要手动配置分辨率

## 🎮 使用示例

### 录制游戏数据

```bash
# 1. 启动游戏
# 2. 运行录制工具
python record_main.py

# 3. 按 Enter 开始录制
# 4. 正常游戏（至少 5 秒）
# 5. 按 ESC 停止并保存

# 高级选项
python record_main.py --save-format frames --frame-step 1 --fps 30
# --save-format: 保存格式 [frames|video|both]
# --frame-step: 帧采样间隔（1=全部保存，5=每5帧保存一次）
# --fps: 目标帧率
```

### AI 控制模式

```bash
# 连接到训练服务器，让 AI 控制游戏
python client_main.py --host 192.168.1.100 --port 9999
```

## 🏗️ 架构

```
游戏画面 → 屏幕捕获 → 网络传输 → 训练服务器
   ↑                                      ↓
键鼠控制 ← 动作执行 ← 网络接收 ← AI 决策
```

### 主要模块

- **`network/game_client.py`** - AI 控制模式，与服务器通信
- **`data_collection/record_gameplay.py`** - 数据录制模式
- **`utils/window_capture.py`** - 智能窗口捕获（新）
- **`cpp_bindings/`** - C++ 高性能模块（屏幕捕获/输入控制）

## 🔧 配置

编辑 `configs/client_config.yaml`：

```yaml
# 屏幕捕获
capture:
  mode: "window"  # 'window' 或 'fullscreen'
  
  window:
    process_name: "WorldOfTanks.exe"  # 游戏进程名
    # 或使用窗口标题：
    # window_title: "World of Tanks"
  
  fps: 30  # 捕获帧率
```

## ⚙️ 依赖

### 必需
- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- pynput（输入监听）

### 可选（提升性能）
- C++ 模块（高性能屏幕捕获）
- pywin32 + psutil（窗口捕获）

## 🐛 故障排查

### 录制时卡顿？

**症状**：录制时画面卡顿，但 CPU/内存占用不高

**原因**：磁盘 I/O 瓶颈（实时保存帧需要持续写入磁盘）

**解决方案**：
1. **诊断**：运行 `python test_disk_performance.py`
2. **优化**：查看 [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md)
3. **快速修复**：
   ```bash
   # 每 5 帧保存一次（节省 80% 磁盘写入）
   python record_main.py --frame-step 5
   ```

**v0.3.1 改进**：已自动启用异步帧保存器，可缓冲 2 秒的帧，大幅减少卡顿！

### 录制目录为空？

**原因 1**：录制时间太短
- **解决**：至少录制 3-5 秒

**原因 2**：找不到游戏窗口
- **解决**：运行 `python test_window_capture.py` 查看所有窗口
- 确认进程名是否正确
- 尝试使用窗口标题：`--window-title "World of Tanks"`

**原因 3**：依赖未安装
- **解决**：`pip install pywin32 psutil mss pynput opencv-python`

更多问题查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 和 [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md)

## 💡 最佳实践

1. **使用 frames 模式**（默认）：实时保存，无内存限制
2. **帧采样**：`--frame-step 5` 节省 80% 存储空间
3. **首次使用先测试**：`python test_capture.py`
4. **长时间录制**：frames 模式支持小时级别录制
5. **使用 C++ 模块**：运行 `build_xmake.bat` 编译以获得最佳性能
6. **游戏设置为窗口化全屏**（可选）：支持窗口捕获模式

## 📊 性能参考

| 配置 | FPS | CPU 占用 | 内存占用（1 分钟） | 磁盘占用（1 分钟） |
|------|-----|---------|------------------|------------------|
| frames + step=1 + 30 FPS | 稳定 30 | ~15% | ~100 MB | ~250 MB |
| frames + step=5 + 30 FPS | 稳定 30 | ~12% | ~50 MB | ~50 MB |
| video + 30 FPS | 稳定 30 | ~20% | ~600 MB | ~50 MB |
| both + 30 FPS | 稳定 30 | ~25% | ~600 MB | ~300 MB |

## 🎓 教程

### 新手入门
1. 阅读 [QUICKSTART_RECORDING.md](QUICKSTART_RECORDING.md)
2. 运行 `python test_capture.py` 诊断系统
3. 尝试录制 10 秒短视频

### 进阶使用
1. 阅读 [WINDOW_CAPTURE_GUIDE.md](WINDOW_CAPTURE_GUIDE.md)
2. 配置 `client_config.yaml`
3. 编译 C++ 模块提升性能

### 故障排查
1. 查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. 运行诊断工具
3. 检查日志输出

## 📝 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解最新功能和改进。

**最新更新 (2025-10-25)**：
- ✨ 新增实时帧保存模式（借鉴参考项目最佳实践）
- ✨ 支持帧采样（`--frame-step`）节省存储
- ✨ 新的 session 数据结构（`meta.json` + `actions.json` + `frames/`）
- ✨ 操作数据与帧号关联
- 🚀 支持长时间录制（小时级别），无内存限制
- 🐛 修复视频编码问题和内存溢出问题
- 📚 更新完整文档

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[MIT License](../LICENSE)

