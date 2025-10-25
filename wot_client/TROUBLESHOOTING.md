# 故障排查指南

## 问题：录制目录为空

### 症状
运行录制后，`data/recordings/recording_YYYYMMDD_HHMMSS/` 目录存在但为空，没有 `gameplay.mp4` 和 `actions.json` 文件。

### 可能原因

#### 1. 录制时间太短
**问题**: 按 Enter 开始录制后立即按 ESC 停止，没有捕获到任何帧。

**解决方法**:
- 至少录制 3-5 秒后再停止
- 等待看到 "录制中..." 的日志信息

#### 2. C++ 模块未编译或加载失败
**问题**: C++ 绑定（`cpp_bindings.pyd`）不存在或无法加载。

**解决方法**:
```bash
# 1. 检查 C++ 模块是否存在
dir cpp_bindings\cpp_bindings.pyd

# 2. 如果不存在，编译 C++ 模块
cd ..
build_xmake.bat

# 3. 运行诊断工具
cd wot_client
python test_capture.py
```

#### 3. Python fallback 依赖未安装
**问题**: C++ 不可用时，Python fallback（mss、pynput）也未安装。

**解决方法**:
```bash
# 激活虚拟环境
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 或手动安装
pip install mss pynput opencv-python numpy loguru
```

#### 4. 屏幕捕获权限不足
**问题**: Windows 屏幕捕获 API 被阻止（如某些游戏反外挂保护）。

**解决方法**:
- 以管理员身份运行
- 检查游戏窗口模式（全屏 vs 窗口化）
- 尝试使用窗口化全屏模式

#### 5. OpenCV VideoWriter 失败
**问题**: 视频写入器无法创建或写入失败。

**解决方法**:
```bash
# 重新安装 OpenCV
pip uninstall opencv-python
pip install opencv-python

# 检查磁盘空间
# 录制会占用大量空间：30秒@30fps ≈ 500MB+
```

## 诊断步骤

### 步骤 1: 运行系统诊断
```bash
cd wot_client
venv\Scripts\activate
python test_capture.py
```

检查输出：
- ✓ 表示测试通过
- ✗ 表示测试失败，查看错误信息

### 步骤 2: 测试屏幕捕获
```bash
python record_main.py --test
```

如果测试通过，说明捕获功能正常。

### 步骤 3: 启用调试日志
编辑 `record_gameplay.py`，修改日志级别：

```python
# 在文件顶部添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

或使用环境变量：
```bash
set LOGURU_LEVEL=DEBUG
python record_main.py
```

### 步骤 4: 检查实时日志
正常录制时应该看到：
```
初始化屏幕捕获模块...
✓ C++ 屏幕捕获模块初始化成功  # 或 Python fallback
初始化输入监听器...
✓ 输入监听器初始化成功
========================================
🎬 录制开始！
  - 按 ESC 键停止录制
  - 目标 FPS: 30
  - C++ 加速: 启用
========================================
录制中... 帧数: 150, 时长: 5.0s, FPS: 30.0
```

如果看到错误信息，根据提示操作。

## 常见错误信息

### `ImportError: DLL load failed`
**原因**: 缺少 MSVC 运行时或依赖库

**解决**:
1. 安装 [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. 重新编译 C++ 模块

### `ModuleNotFoundError: No module named 'mss'`
**原因**: Python fallback 依赖未安装

**解决**:
```bash
pip install mss pynput
```

### `RuntimeError: Failed to open video writer`
**原因**: OpenCV 无法创建视频文件

**解决**:
1. 检查磁盘空间
2. 检查输出目录权限
3. 尝试不同的编解码器（修改代码中的 `fourcc`）

### `❌ 录制失败：没有捕获到任何帧！`
**原因**: 见上述"可能原因"部分

**解决**: 按照上面的诊断步骤逐一排查

## 性能问题

### 录制卡顿或 FPS 低
1. **降低录制分辨率**: 修改 `record_gameplay.py` 中的 `1920, 1080`
2. **降低 FPS**: 使用 `--fps 15` 或 `--fps 20`
3. **确保使用 C++ 模块**: Python fallback 性能较低
4. **关闭其他应用**: 减少系统负载

### 内存不足
录制时间过长会占用大量内存（所有帧都存在内存中）。

**解决**:
- 分段录制，每次不超过 1-2 分钟
- 或修改代码，改为实时写入视频（更复杂）

## 仍然无法解决？

1. **查看完整日志**: 保存控制台所有输出
2. **提供环境信息**:
   ```bash
   python test_capture.py > diagnostic.txt 2>&1
   ```
3. **检查文件权限**: 确保对 `data/recordings/` 有写权限
4. **杀毒软件**: 某些杀毒软件会阻止屏幕捕获或输入模拟

## 调试技巧

### 测试最小化代码
创建一个简单的测试脚本：

```python
# test_minimal.py
import sys
sys.path.append('.')

try:
    from cpp_bindings import ScreenCapture
    sc = ScreenCapture(1920, 1080)
    buffer = sc.Capture()
    print(f"✓ 捕获成功: {len(buffer)} 字节")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
```

### 逐步验证
1. 导入模块
2. 初始化捕获器
3. 捕获一帧
4. 保存为图片
5. 循环捕获
6. 写入视频

在每一步检查是否成功。

## 成功标志

录制成功后应该看到：
```
========================================
✓ 录制成功保存到: data/recordings/recording_20251025_154556
  - 总帧数: 150
  - 时长: 5.00s
  - 平均 FPS: 30.00
========================================
```

目录结构：
```
data/recordings/recording_20251025_154556/
├── gameplay.mp4      # 视频文件（应有几百 KB 到 MB）
└── actions.json      # 操作记录（包含按键、鼠标数据）
```

