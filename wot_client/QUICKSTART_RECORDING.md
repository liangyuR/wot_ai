# 快速开始 - 游戏录制

## 🚀 5 分钟快速开始

### 1. 首次设置

```bash
cd wot_client

# 方式 1: 使用批处理脚本（推荐）
start_recording.bat

# 方式 2: 手动设置
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 系统检查（推荐）

```bash
# 运行诊断工具
python test_capture.py
```

确保看到：
- ✓ C++ 屏幕捕获模块初始化成功（或 Python fallback）
- ✓ mss 模块导入成功
- ✓ OpenCV 导入成功

### 3. 测试屏幕捕获

```bash
# 快速测试（不实际录制）
python record_main.py --test
```

### 4. 开始录制

```bash
# 启动录制工具
python record_main.py

# 或使用批处理脚本
start_recording.bat
```

**操作流程**:
1. 启动《坦克世界》并进入战斗
2. 回到录制工具，按 Enter 开始
3. 正常游戏（至少 5 秒）
4. 按 **ESC** 键停止并保存

### 5. 查看结果

```bash
cd data\recordings\recording_YYYYMMDD_HHMMSS
dir

# 应该看到：
# gameplay.mp4     - 游戏视频
# actions.json     - 操作记录
```

## 📝 命令行选项

```bash
# 指定输出目录
python record_main.py --output my_recordings

# 设置 FPS（默认 30）
python record_main.py --fps 20

# 测试模式
python record_main.py --test
```

## ⚙️ 配置优化

### 低配电脑
```bash
# 降低 FPS 和分辨率
python record_main.py --fps 15
```

修改 `record_gameplay.py` 中的分辨率：
```python
# 从 1920x1080 改为 1280x720
self.screen_capture_ = ScreenCapture(1280, 720)
self.monitor_ = {"top": 0, "left": 0, "width": 1280, "height": 720}
```

### 高性能录制
确保使用 C++ 模块（显著提升性能）：
```bash
cd ..
build_xmake.bat
```

## 🔍 故障排查

### 录制目录为空？

**原因 1**: 录制时间太短
- **解决**: 至少录制 3-5 秒

**原因 2**: 模块初始化失败
- **解决**: 运行 `python test_capture.py` 诊断

**原因 3**: 依赖未安装
- **解决**: 
  ```bash
  pip install mss pynput opencv-python numpy loguru
  ```

### 查看详细错误
所有错误信息会显示在控制台，包括：
- 初始化状态
- 实时录制进度
- 保存结果

### 更多帮助
查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 获取详细故障排查指南。

## 📊 预期输出

### 成功的录制日志
```
========================================
🎮 World of Tanks - 游戏录制工具
========================================
输出目录: data/recordings
录制 FPS: 30
C++ 加速: 可用
...
初始化屏幕捕获模块...
✓ C++ 屏幕捕获模块初始化成功
初始化输入监听器...
✓ 输入监听器初始化成功

按 Enter 键继续...

========================================
🎬 录制开始！
  - 按 ESC 键停止录制
  - 目标 FPS: 30
  - C++ 加速: 启用
========================================
录制中... 帧数: 150, 时长: 5.0s, FPS: 30.0
录制中... 帧数: 300, 时长: 10.0s, FPS: 30.0
Stopping recording...
Saving video...
Writing frame 0/300...
Writing frame 100/300...
Writing frame 200/300...
✓ Video saved: data/recordings/recording_20251025_154556/gameplay.mp4
Saving action data...
✓ Actions saved: data/recordings/recording_20251025_154556/actions.json
========================================
✓ 录制成功保存到: data/recordings/recording_20251025_154556
  - 总帧数: 300
  - 时长: 10.00s
  - 平均 FPS: 30.00
========================================
```

### 文件大小参考
- **5 秒 @ 30 FPS**: ~50-100 MB
- **30 秒 @ 30 FPS**: ~300-600 MB
- **1 分钟 @ 30 FPS**: ~600-1200 MB

## 💡 最佳实践

1. **首次使用先测试**: 运行 `python test_capture.py`
2. **分段录制**: 每次 30 秒到 1 分钟，避免内存不足
3. **使用 C++ 加速**: 性能提升 2-5 倍
4. **管理员权限**: 某些情况需要以管理员身份运行
5. **窗口化全屏**: 比完全全屏更稳定
6. **监控磁盘空间**: 录制消耗大量空间

## 🎯 下一步

录制数据后，可以用于：
1. **模仿学习**: 训练 AI 模仿人类玩家
2. **数据分析**: 分析游戏行为模式
3. **测试 AI**: 对比 AI 和人类表现

查看 `wot_server/` 了解如何训练 AI 模型。

