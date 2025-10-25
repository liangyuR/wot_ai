# 窗口捕获使用指南

## 📖 功能说明

窗口捕获是一种智能的屏幕捕获方式，可以：
- ✅ **自动定位游戏窗口**：通过进程名或窗口标题自动找到游戏
- ✅ **自动跟随窗口**：游戏窗口移动时自动跟随
- ✅ **自适应分辨率**：自动适配窗口大小
- ✅ **只捕获游戏内容**：不包含其他桌面内容
- ✅ **支持多显示器**：无论游戏在哪个屏幕

与传统的 ROI（感兴趣区域）配置相比，窗口捕获**无需手动配置坐标**，更智能、更灵活。

## 🚀 快速开始

### 1. 安装依赖

```bash
cd wot_client
venv\Scripts\activate
pip install pywin32 psutil
```

### 2. 查看所有窗口（可选）

运行测试工具查看系统中的所有窗口：

```bash
python test_window_capture.py
```

这会列出所有可见窗口及其进程名，帮助你确定正确的进程名或窗口标题。

### 3. 测试窗口捕获

在开始正式录制前，先测试窗口捕获：

```bash
# 测试通过进程名捕获
python record_main.py --test --mode window --process WorldOfTanks.exe

# 或测试通过窗口标题捕获
python record_main.py --test --mode window --window-title "World of Tanks"
```

### 4. 开始录制

```bash
# 使用进程名（推荐）
python record_main.py --mode window --process WorldOfTanks.exe

# 或使用窗口标题
python record_main.py --mode window --window-title "World of Tanks"
```

## 🎯 两种捕获模式对比

### Window 模式（推荐）

```bash
python record_main.py --mode window --process WorldOfTanks.exe
```

**优点**：
- 自动定位游戏窗口
- 窗口移动时自动跟随
- 只捕获游戏内容
- 支持窗口化和窗口化全屏

**缺点**：
- 需要额外依赖（pywin32、psutil）
- 不支持完全全屏模式（DirectX 独占）

**适用场景**：
- 游戏使用窗口化或窗口化全屏模式
- 多任务环境
- 需要精确捕获游戏内容

### Fullscreen 模式

```bash
python record_main.py --mode fullscreen
```

**优点**：
- 无需额外依赖
- 支持完全全屏模式
- 实现简单

**缺点**：
- 捕获整个屏幕（包含任务栏、其他窗口等）
- 需要手动配置分辨率
- 不能自动跟随窗口

**适用场景**：
- 游戏使用完全全屏模式
- 屏幕只有游戏内容
- Window 模式不可用时

## 📝 配置文件

编辑 `configs/client_config.yaml`：

```yaml
# 屏幕捕获
capture:
  # 捕获模式：'window' 或 'fullscreen'
  mode: "window"  # 推荐使用 window 模式
  
  # 窗口捕获配置（mode: window 时使用）
  window:
    # 通过进程名称查找窗口（推荐）
    process_name: "WorldOfTanks.exe"
    
    # 或通过窗口标题查找（备选方案）
    # window_title: "World of Tanks"
    
    # 是否使用部分匹配
    partial_match: true
    
    # 如果窗口未找到，是否降级到全屏模式
    fallback_to_fullscreen: true
  
  # 全屏捕获配置（mode: fullscreen 时使用）
  fullscreen:
    width: 1920
    height: 1080
  
  # 通用配置
  fps: 30
```

## 🔍 查找进程名或窗口标题

### 方法 1: 使用测试工具

```bash
python test_window_capture.py
```

会列出所有窗口，找到游戏对应的进程名。

### 方法 2: 使用任务管理器

1. 打开任务管理器（Ctrl+Shift+Esc）
2. 切换到"详细信息"标签
3. 找到游戏进程，查看"名称"列

### 方法 3: 常见游戏进程名

| 游戏 | 进程名 | 窗口标题 |
|------|--------|----------|
| 坦克世界 | WorldOfTanks.exe | World of Tanks |
| 英雄联盟 | League of Legends.exe | League of Legends |
| CS:GO | csgo.exe | Counter-Strike |
| Dota 2 | dota2.exe | Dota 2 |

## 🐛 故障排查

### 问题 1: 找不到游戏窗口

**症状**：
```
✗ 未找到目标窗口
```

**解决方法**：

1. **确保游戏已启动并且窗口可见**（未最小化）

2. **运行测试工具查看所有窗口**：
   ```bash
   python test_window_capture.py
   ```

3. **检查进程名是否正确**：
   - 进程名区分大小写
   - 必须包含 `.exe` 后缀
   - 使用 `--window-title` 作为备选方案

4. **尝试使用部分匹配**（默认已启用）

### 问题 2: 游戏使用完全全屏模式

**症状**：
窗口捕获无法获取全屏游戏画面（DirectX 独占模式）

**解决方法**：

1. **切换游戏到窗口化全屏模式**（推荐）
   - 在游戏设置中找到"显示模式"
   - 选择"窗口化全屏"或"无边框窗口"

2. **使用 fullscreen 模式**：
   ```bash
   python record_main.py --mode fullscreen
   ```

### 问题 3: pywin32 安装失败

**症状**：
```
ImportError: No module named 'win32gui'
```

**解决方法**：

```bash
# 方法 1: 使用 pip
pip install pywin32

# 方法 2: 运行安装脚本
python venv\Scripts\pywin32_postinstall.py -install

# 方法 3: 重新安装
pip uninstall pywin32
pip install pywin32
```

### 问题 4: 捕获帧率低

**可能原因**：
- 窗口捕获比全屏捕获稍慢（但更精确）
- 游戏分辨率太高

**解决方法**：

1. **降低录制 FPS**：
   ```bash
   python record_main.py --mode window --fps 20
   ```

2. **降低游戏分辨率**（在游戏设置中）

3. **使用 C++ 加速模块**（如果可用）

### 问题 5: 捕获的画面是黑色或空白

**可能原因**：
- 游戏使用了硬件加速或 GPU 保护
- 某些反外挂系统阻止屏幕捕获

**解决方法**：

1. **以管理员身份运行**录制工具

2. **在游戏设置中禁用硬件加速**（如果有）

3. **使用 fullscreen 模式**作为备选

4. **检查是否有安全软件阻止屏幕捕获**

## 📊 性能对比

| 模式 | 性能 | 精确度 | 灵活性 |
|------|------|--------|--------|
| Window + pywin32 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Fullscreen + C++ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Fullscreen + mss | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

推荐：**Window 模式** - 最佳的精确度和灵活性平衡。

## 💡 最佳实践

1. **优先使用 Window 模式**：
   ```bash
   python record_main.py --mode window --process WorldOfTanks.exe
   ```

2. **游戏设置为窗口化全屏**：
   - 在游戏中设置显示模式为"窗口化全屏"
   - 这样既保持全屏体验，又能被窗口捕获

3. **首次使用先测试**：
   ```bash
   # 列出所有窗口
   python test_window_capture.py
   
   # 测试捕获
   python test_window_capture.py  # 然后选择交互式测试
   ```

4. **使用进程名而不是窗口标题**：
   - 进程名更稳定（窗口标题可能随游戏状态变化）
   - 进程名匹配更快

5. **配置降级策略**：
   ```yaml
   window:
     fallback_to_fullscreen: true  # 窗口未找到时自动降级
   ```

## 🎮 实际使用示例

### 示例 1: 录制《坦克世界》

```bash
# 1. 启动游戏，设置为窗口化全屏
# 2. 运行录制工具
python record_main.py --mode window --process WorldOfTanks.exe --fps 30

# 3. 按 Enter 开始录制
# 4. 正常游戏
# 5. 按 ESC 停止
```

### 示例 2: 测试任意窗口

```bash
# 测试 Chrome 浏览器
python test_window_capture.py
# 然后选择 2，输入 "chrome"

# 或直接录制 Chrome
python record_main.py --mode window --window-title "Chrome" --fps 15
```

### 示例 3: 批处理脚本

创建 `record_wot.bat`：

```batch
@echo off
cd wot_client
call venv\Scripts\activate
python record_main.py --mode window --process WorldOfTanks.exe --fps 30
pause
```

## 📚 相关文档

- [快速开始](QUICKSTART_RECORDING.md) - 5 分钟入门指南
- [故障排查](TROUBLESHOOTING.md) - 详细的问题解决方案
- [配置文件说明](configs/client_config.yaml) - 配置选项详解

## ❓ 常见问题

**Q: Window 模式和 Fullscreen 模式有什么区别？**

A: Window 模式通过进程名或窗口标题自动定位游戏窗口，只捕获游戏内容。Fullscreen 模式捕获整个屏幕。推荐使用 Window 模式。

**Q: 我的游戏是完全全屏，Window 模式不工作怎么办？**

A: 将游戏设置改为"窗口化全屏"模式，或使用 Fullscreen 模式录制。

**Q: 窗口捕获的性能如何？**

A: 略低于 C++ 全屏捕获，但明显优于 Python fallback。对于 30 FPS 录制完全足够。

**Q: 可以同时录制多个游戏窗口吗？**

A: 当前版本不支持。需要运行多个录制实例，分别指定不同的进程名。

**Q: 支持哪些 Windows 版本？**

A: Windows 7 及以上版本。推荐 Windows 10/11。

