# 更新日志

## [2025-10-25] v0.3.1 - 异步帧保存（性能优化）

### ✨ 新功能

#### 1. 异步帧保存器
- **独立线程保存帧**，主循环不再阻塞
- **队列缓冲 60 帧**（2 秒），平滑性能波动
- 添加到队列仅需 <1ms，不影响捕获

**工作原理**：
```
捕获帧 → 添加到队列（<1ms）→ 继续捕获
              ↓
        后台线程保存到磁盘
```

**效果**：
- 同步保存：每帧阻塞 15-20ms → 异步保存：<1ms
- 大幅减少录制卡顿
- 对慢速磁盘更友好

#### 2. 实时性能监控
- 显示队列状态：`队列: 3/60`
- 统计丢帧数：`已保存=287, 丢弃=0`
- 方便诊断磁盘性能问题

#### 3. 磁盘性能测试工具
新增 `test_disk_performance.py`：
- 测试磁盘写入速度
- 测试不同 JPEG 质量的性能
- 提供优化建议

### 🐛 Bug 修复

#### 1. 录制卡顿问题
- **问题**：CPU/内存占用不高，但画面卡顿
- **原因**：磁盘 I/O 瓶颈（同步保存阻塞主循环）
- **解决**：异步帧保存器

### 📚 新增文档

1. **`PERFORMANCE_GUIDE.md`** - 性能优化完整指南
   - 问题诊断
   - 解决方案（5 种方法）
   - 配置推荐
   - 故障排查

2. **`async_frame_saver.py`** - 异步帧保存器实现
   - 队列管理
   - 独立线程
   - 统计信息

3. **`test_disk_performance.py`** - 磁盘性能诊断工具
   - 测试写入速度
   - 测试不同质量
   - 提供建议

### 📈 性能改进

| 指标 | v0.3.0（同步） | v0.3.1（异步） | 改进 |
|------|--------------|--------------|------|
| 主循环阻塞 | 15-20ms/帧 | <1ms/帧 | **95%** ⬇ |
| 帧率稳定性 | 波动大 | 平滑 | ✓ |
| 磁盘要求 | 严格 | 宽松 | ✓ |
| 丢帧风险 | 高 | 低 | ✓ |

### 💡 使用建议

**如果录制卡顿**：
1. 运行诊断：`python test_disk_performance.py`
2. 查看指南：`PERFORMANCE_GUIDE.md`
3. 快速修复：`python record_main.py --frame-step 5`

**监控性能**：
```
录制中... 帧数: 150, 时长: 5.0s, FPS: 30.0, 队列: 3/60
                                              ↑
                                      队列状态（正常）
```

### 🔄 向后兼容

- 自动检测异步保存器是否可用
- 不可用时回退到同步保存
- 无需修改现有代码

### 🎯 技术细节

**队列设计**：
- 大小：60 帧（2 秒 @ 30 FPS）
- 当队列满时丢帧并警告
- 停止录制时等待队列清空

**线程安全**：
- 使用 `queue.Queue`（线程安全）
- 使用 `put_nowait` 避免阻塞
- 优雅关闭（`queue.join()`）

---

## [2025-10-25] v0.3.0 - 实时帧保存 + Session 结构

### ✨ 新功能

#### 1. 实时帧保存模式（推荐）
- **边录制边保存 JPEG 帧到磁盘**
  - 无内存限制，支持长时间录制（小时级别）
  - 每帧实时写入磁盘（JPEG 质量 95%）
  - 不再需要将所有帧存储在内存中
  
- **可配置保存格式**：
  - `frames`（默认）：只保存帧，推荐
  - `video`：编码为视频，占用内存
  - `both`：同时保存帧和视频
  
- **帧采样支持**：
  - `--frame-step 1`：保存所有帧
  - `--frame-step 5`：每 5 帧保存一次，节省 80% 存储
  - actions.json 仍然记录所有帧的操作

#### 2. 新的 Session 数据结构
- **结构化目录**：
  ```
  data/sessions/
  └── session_YYYYMMDD_HHMMSS/
      ├── meta.json         # 会话元数据
      ├── actions.json      # 操作记录（关联帧号）
      ├── frames/           # JPEG 帧
      │   ├── frame_000000.jpg
      │   └── ...
      └── gameplay.avi      # 视频（可选）
  ```

- **meta.json**：包含 fps、分辨率、时长等元数据
- **actions.json**：每条记录包含 `frame` 字段，直接关联到 `frames/frame_XXXXXX.jpg`

#### 3. 新增命令行参数
- `--save-format [frames|video|both]`：选择保存格式
- `--frame-step N`：帧采样间隔（N=1 保存所有帧）
- `--output data/sessions`：输出目录（默认）

### 🐛 Bug 修复

#### 1. 内存溢出问题
- **问题**：长时间录制导致内存不足
- **解决**：frames 模式实时保存到磁盘

#### 2. 视频编码失败
- **问题**：某些编解码器不兼容，视频无法播放或为空
- **解决**：
  - 简化为单一 XVID 编解码器
  - frames 模式作为主要方式，视频为可选
  - 后期可用 ffmpeg 转换帧为视频

#### 3. 竞态条件
- **问题**：`stopRecording()` 被调用两次（ESC 键 + finally 块）
- **解决**：添加 `saved_` 标志防止重复保存

### 📈 改进

#### 1. 性能优化
- **内存占用降低 85%**（frames 模式 vs video 模式）
- **支持长时间录制**：1 小时+ 无压力
- **磁盘写入优化**：JPEG 质量 95%，1920x1080 约 200-300 KB/帧

#### 2. 数据加载更灵活
- 可选择性加载帧，无需全部加载到内存
- 支持快速跳帧和随机访问
- 更适合大规模训练（可使用 DataLoader 流式加载）

#### 3. 存储空间优化
- 帧采样：`--frame-step 5` 可节省 80% 空间
- 1 分钟 30 FPS：
  - step=1：约 360 MB
  - step=5：约 72 MB

### 📚 新增文档

1. **`DATA_FORMAT.md`** - 数据格式完整说明
   - 目录结构详解
   - meta.json 和 actions.json 格式
   - 数据加载示例
   - 最佳实践

2. **`test_session_format.py`** - Session 数据验证工具
   - 验证数据完整性
   - 检查文件格式
   - 统计信息

3. **更新 `README.md`**
   - 新增保存模式说明
   - 更新使用示例
   - 更新性能参考

### 🔄 向后兼容

- 保留 video 模式（不推荐）
- 命令行参数向后兼容
- 默认使用新的 frames 模式

### 🚀 迁移指南

#### 从 v0.2.0 迁移到 v0.3.0

**无需修改任何代码！** 默认配置已经是最佳实践。

**旧版本录制方式（仍然支持）**：
```bash
python record_main.py --save-format video
# 输出: data/sessions/session_YYYYMMDD_HHMMSS/gameplay.avi
```

**新版本推荐方式**：
```bash
python record_main.py
# 输出: data/sessions/session_YYYYMMDD_HHMMSS/frames/*.jpg + meta.json + actions.json
```

**数据格式变化**：
- 输出目录：`data/recordings/` → `data/sessions/`
- actions.json：使用 `frame` 代替 `timestamps` 作为主键
- 新增 `meta.json` 文件

### 📊 性能对比

| 模式 | 内存占用（1分钟） | 磁盘占用（1分钟） | 最大录制时长 | 推荐度 |
|------|------------------|------------------|------------|-------|
| frames (step=1) | ~100 MB | ~360 MB | 无限制 | ⭐⭐⭐⭐⭐ |
| frames (step=5) | ~50 MB | ~72 MB | 无限制 | ⭐⭐⭐⭐⭐ |
| video | ~600 MB | ~50 MB | ~10 分钟 | ⭐⭐ |
| both | ~600 MB | ~400 MB | ~10 分钟 | ⭐ |

### 💡 最佳实践

1. **使用 frames 模式**（默认）- 实时保存，无内存限制
2. **合理设置 frame_step**：
   - 训练模仿学习：`step=1` 或 `step=2`
   - 数据分析：`step=5`
3. **后期需要视频时使用 ffmpeg 转换**：
   ```bash
   ffmpeg -framerate 30 -i frames/frame_%06d.jpg -c:v libx264 output.mp4
   ```

### 🎯 设计灵感

本版本的改进借鉴了以下项目的最佳实践：
- [record-gameplay](https://github.com/Oughie/record-gameplay) - 实时帧保存、session 结构

感谢开源社区！

### 🐛 已知问题

- 无

---

## [2024-10-25] - 窗口捕获功能 + Bug 修复

### ✨ 新功能

#### 1. 智能窗口捕获
- **通过进程名自动定位游戏窗口**
  - 示例：`python record_main.py --mode window --process WorldOfTanks.exe`
  - 无需手动配置 ROI 坐标
  
- **通过窗口标题查找**
  - 备选方案：`python record_main.py --mode window --window-title "World of Tanks"`
  - 支持部分匹配
  
- **自动跟随窗口**
  - 游戏窗口移动时自动跟随
  - 支持动态调整窗口大小
  
- **多显示器支持**
  - 无论游戏在哪个屏幕都能正确捕获

#### 2. 新增工具

- **`test_window_capture.py`** - 窗口捕获测试工具
  - 列出所有可见窗口
  - 交互式测试捕获功能
  - 帮助确定正确的进程名/窗口标题
  
- **`test_capture.py`** - 系统诊断工具（之前版本）
  - 测试 C++ 绑定
  - 测试 Python fallback
  - 显示系统信息

#### 3. 配置文件增强

更新 `configs/client_config.yaml`：
- 新增 `capture.mode` 选项（window/fullscreen）
- 新增 `capture.window` 配置块
- 保留 `capture.fullscreen` 兼容旧配置
- 标记 `capture.roi` 为已弃用

### 🐛 Bug 修复

#### 1. 录制目录为空问题
- **修复空列表访问崩溃**：`self.timestamps_[-1]` 在空列表时会抛出 `IndexError`
- **添加数据检查**：录制前检查是否捕获到任何帧
- **详细错误提示**：失败时显示可能原因和调试信息

#### 2. 异常处理改进
- 捕获过程中的错误现在会被正确处理而不是崩溃
- 添加 try-catch 到所有关键函数
- 保存失败时显示详细错误和堆栈跟踪

#### 3. 动态分辨率支持
- 视频保存时自动检测帧的实际分辨率
- 支持窗口大小变化（窗口模式下）
- 不再硬编码 1920x1080

### 📈 改进

#### 1. 更详细的日志
- 初始化时显示每个模块的状态
- 录制时每 5 秒显示进度（帧数、时长、FPS）
- 保存时每 100 帧显示进度
- 所有日志使用中文，更友好

#### 2. 增强的启动脚本
`start_recording.bat` 现在提供交互式菜单：
1. 开始录制游戏
2. 测试屏幕捕获
3. 运行系统诊断
4. 退出

#### 3. 命令行参数
新增参数：
- `--mode window|fullscreen` - 选择捕获模式
- `--process PROCESS_NAME` - 指定进程名
- `--window-title TITLE` - 指定窗口标题
- `--test` - 测试模式（不实际录制）

### 📚 新增文档

1. **`WINDOW_CAPTURE_GUIDE.md`** - 窗口捕获完整指南
   - 功能说明
   - 快速开始
   - 两种模式对比
   - 故障排查
   - 最佳实践

2. **`TROUBLESHOOTING.md`** - 详细故障排查指南
   - 录制目录为空问题
   - 依赖安装问题
   - 性能问题
   - 调试技巧

3. **`QUICKSTART_RECORDING.md`** - 5 分钟快速开始
   - 首次设置
   - 测试
   - 开始录制

### 📦 依赖更新

`requirements.txt` 新增：
- `psutil>=5.9.0` - 进程管理（窗口捕获需要）

现有依赖：
- `pywin32>=305` - Windows API（窗口捕获需要）
- `mss>=9.0.0` - 全屏捕获 fallback
- `pynput>=1.7.6` - 输入监听

### 🔄 向后兼容

- 保留所有现有功能
- 默认使用窗口模式，但可降级到全屏模式
- 旧的配置文件仍然有效
- 命令行参数向后兼容

### 🚀 使用示例

#### 之前的方式（仍然支持）
```bash
python record_main.py
# 捕获整个屏幕 1920x1080
```

#### 新的推荐方式
```bash
# 窗口模式 - 自动定位游戏
python record_main.py --mode window --process WorldOfTanks.exe

# 或使用窗口标题
python record_main.py --mode window --window-title "World of Tanks"

# 测试窗口捕获
python test_window_capture.py
```

### ⚠️ 已知限制

1. **窗口模式不支持 DirectX 独占全屏**
   - 解决方案：将游戏设置为窗口化全屏
   - 或使用 fullscreen 模式

2. **某些反外挂游戏可能阻止窗口捕获**
   - 解决方案：以管理员身份运行
   - 或使用 fullscreen 模式

3. **窗口捕获性能略低于 C++ 全屏捕获**
   - 但对 30 FPS 录制完全足够
   - 可降低 FPS 到 20 或 15

### 🎯 下一步计划

- [ ] 支持多窗口同时录制
- [ ] 添加实时预览窗口
- [ ] 支持热键开始/停止录制
- [ ] 添加视频压缩选项
- [ ] 支持直接写入磁盘（减少内存占用）
- [ ] GPU 加速捕获（NVENC）

### 💡 迁移指南

如果你之前使用全屏模式，现在想切换到窗口模式：

1. **确保游戏已启动**

2. **查找游戏进程名**：
   ```bash
   python test_window_capture.py
   ```

3. **更新配置文件**（可选）：
   ```yaml
   capture:
     mode: "window"
     window:
       process_name: "YourGame.exe"
   ```

4. **开始录制**：
   ```bash
   python record_main.py --mode window --process YourGame.exe
   ```

### 🙏 贡献者

感谢用户反馈帮助我们发现并修复录制目录为空的问题！

---

## 版本历史

### [2024-10-25] v0.2.0
- 添加窗口捕获功能
- 修复录制空目录 bug
- 大量日志和文档改进

### [2024-XX-XX] v0.1.0
- 初始版本
- 基础屏幕捕获和录制功能

