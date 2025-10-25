# 安装指南

## 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 3060 或更高（推荐 RTX 4090/5090）
- **内存**: 16GB RAM 最低，32GB 推荐
- **存储**: 50GB 可用空间

### 软件要求
- **操作系统**: Windows 10/11 (64位)
- **Python**: 3.10 或更高
- **CUDA**: 11.8 或更高（用于 PyTorch GPU 加速）
- **Visual Studio**: 2019 或 2022（带 C++ 支持）
- **World of Tanks**: 最新版本客户端

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/world_of_tanks_ai.git
cd world_of_tanks_ai
```

### 2. 安装 Visual Studio C++ 工具

1. 下载 [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/)
2. 安装时选择 "Desktop development with C++"
3. 确保包含 CMake 和 Windows 10 SDK

### 3. 安装 CUDA 和 cuDNN

1. 下载 [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-downloads)
2. 下载 [cuDNN](https://developer.nvidia.com/cudnn)
3. 按照官方指南安装

### 4. 运行构建脚本

**Windows:**
```bash
build.bat
```

这将自动：
- 创建 Python 虚拟环境
- 安装 Python 依赖
- 编译 C++ 模块
- 安装 Python bindings

### 5. 验证安装

```python
# 测试 Python 导入
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from cpp_bindings import ScreenCapture; print('C++ bindings OK')"
```

## 手动安装（如果自动脚本失败）

### Python 环境

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### C++ 模块

```bash
cd cpp
mkdir build
cd build

# 配置
cmake .. -G "Visual Studio 17 2022" -A x64

# 编译
cmake --build . --config Release

# 安装
cmake --install . --config Release
```

## 常见问题

### Q1: CMake 找不到 Python

**解决方案**: 设置 Python 路径
```bash
cmake .. -DPython3_ROOT_DIR="C:\Python310"
```

### Q2: CUDA 版本不匹配

**解决方案**: 安装匹配的 PyTorch 版本
```bash
# 例如 CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q3: pybind11 找不到

**解决方案**:
```bash
pip install pybind11
```

### Q4: 编译时出现链接错误

**解决方案**: 确保安装了 Windows SDK
- 打开 Visual Studio Installer
- 修改安装
- 选择 "Windows 10 SDK"

## 测试安装

运行测试脚本验证所有组件：

```bash
python python/tests/test_installation.py
```

## 下一步

安装完成后，参考：
- [快速开始](QUICKSTART.md) - 运行第一个示例
- [训练指南](TRAINING.md) - 训练自己的 AI
- [配置说明](CONFIGURATION.md) - 配置参数详解

