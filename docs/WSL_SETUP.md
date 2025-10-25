# WSL 环境配置指南

## WSL 特点

WSL (Windows Subsystem for Linux) 是在 Windows 上运行 Linux 的子系统，有一些特殊考虑：

### ✅ WSL 适合做什么
- Python 代码开发和测试
- Linux 服务器代码开发
- 跨平台兼容性测试

### ⚠️ WSL 不适合做什么
- 屏幕捕获（需要访问 Windows 显示）
- GPU 训练（WSL2 支持，但配置复杂）
- 生产环境部署

---

## 推荐开发流程

### 方案 A：混合开发（推荐）

```
Windows 侧                WSL 侧
├── C++ 模块编译         ├── Python 开发
├── 游戏运行             ├── 代码测试
└── 屏幕捕获             └── Linux 兼容性验证
```

**步骤**：

1. **Windows 端编译 C++ 模块**
```powershell
# 在 Windows PowerShell 中
cd d:\projects\world_of_tanks
.\build_xmake.bat
```

2. **WSL 端开发 Python 代码**
```bash
# 在 WSL 中
cd /mnt/d/projects/world_of_tanks
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 开发和测试
python python/tests/test_installation.py
```

---

## WSL 完整配置

### 1. 安装必要的包

```bash
sudo apt update
sudo apt install -y \
    python3.10 python3-pip python3-venv \
    build-essential \
    libx11-dev libxext-dev libxtst-dev
```

### 2. 安装 xmake（可选）

```bash
# 方法 1: 官方脚本
bash <(curl -fsSL https://xmake.io/shget.text)

# 方法 2: 从 release 下载
wget https://github.com/xmake-io/xmake/releases/download/v2.8.5/xmake-v2.8.5.gz.run
bash xmake-v2.8.5.gz.run --prefix=/usr/local
```

### 3. 配置 Python 环境

```bash
cd /mnt/d/projects/world_of_tanks

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 4. 构建 C++ 模块（可选）

```bash
# 如果需要在 WSL 中构建 C++ 模块
./build_xmake.sh

# 注意：屏幕捕获功能在 WSL 中可能无法工作
# 建议使用 Python fallback
```

---

## 常见问题

### ❌ 问题 1: pip install 报错 "externally-managed-environment"

**症状**:
```
error: externally-managed-environment
```

**解决方案**:
```bash
# 使用虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 或使用 --break-system-packages（不推荐）
pip install --break-system-packages -r requirements.txt
```

### ❌ 问题 2: X11 库编译错误

**症状**:
```
fatal error: X11/Xlib.h: No such file or directory
```

**解决方案**:
```bash
# 安装 X11 开发库
sudo apt install -y libx11-dev libxext-dev libxtst-dev

# 重新构建
./build_xmake.sh
```

### ❌ 问题 3: 屏幕捕获不工作

**原因**: WSL 无法直接访问 Windows 图形界面

**解决方案**:

**选项 A: 使用 Python fallback（推荐）**
```python
# 代码会自动检测并使用 mss 库
# 无需 C++ 模块
```

**选项 B: 在 Windows 端运行游戏客户端**
```bash
# WSL 端：运行训练服务器
python python/network/training_server.py --host 0.0.0.0

# Windows 端：运行游戏客户端
python python/network/game_client.py --host localhost
```

### ❌ 问题 4: GPU 不可用

**WSL2 GPU 支持**:

```bash
# 检查 GPU
nvidia-smi

# 如果不可用，需要：
# 1. 更新 Windows 到最新版本
# 2. 安装 NVIDIA WSL 驱动
# 3. 更新 WSL2 内核
```

**详细步骤**: https://docs.microsoft.com/windows/wsl/tutorials/gpu-compute

---

## 推荐配置

### 配置 1: 纯 Python 开发（最简单）

```bash
# WSL 中
cd /mnt/d/projects/world_of_tanks
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 不编译 C++ 模块，使用 Python fallback
python python/tests/test_installation.py
```

### 配置 2: 完整开发环境

```bash
# 安装所有依赖
sudo apt install -y \
    python3.10 python3-pip python3-venv \
    build-essential \
    libx11-dev libxext-dev libxtst-dev \
    cmake ninja-build

# 构建 C++ 模块
./build_xmake.sh

# 测试
python python/tests/test_installation.py
```

### 配置 3: 训练服务器（WSL2 + GPU）

```bash
# 确保 WSL2 GPU 支持已启用
nvidia-smi

# 安装 CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 文件系统注意事项

### Windows 文件系统性能

```bash
# ⚠️ 慢：在 /mnt/c 或 /mnt/d 下工作
cd /mnt/d/projects/world_of_tanks

# ✅ 快：在 WSL 原生文件系统工作
cd ~
cp -r /mnt/d/projects/world_of_tanks ~/wot_ai
cd ~/wot_ai
```

**性能对比**:
- Windows 文件系统 (/mnt/d): ~100 MB/s
- WSL 原生文件系统 (~): ~1000 MB/s

**建议**: 
- 开发时可以在 Windows 文件系统（方便 IDE 访问）
- 训练时复制到 WSL 原生文件系统（性能更好）

---

## VSCode 集成

### 安装 WSL 扩展

1. 安装 "Remote - WSL" 扩展
2. 在 WSL 中打开项目：
```bash
cd /mnt/d/projects/world_of_tanks
code .
```

### 配置 Python 解释器

1. Ctrl+Shift+P
2. "Python: Select Interpreter"
3. 选择 WSL 虚拟环境: `~/venv/bin/python`

---

## 最佳实践

### 开发工作流

```bash
# 1. Windows 端编辑代码
#    使用 VSCode/PyCharm 等 IDE

# 2. WSL 端测试
cd /mnt/d/projects/world_of_tanks
source venv/bin/activate
python python/tests/test_installation.py

# 3. 提交前检查
black python/  # 代码格式化
flake8 python/  # 代码检查
pytest python/tests/  # 运行测试
```

### 同步到真实 Linux 服务器

```bash
# 在 WSL 中测试通过后
rsync -avz --exclude='venv' --exclude='__pycache__' \
    /mnt/d/projects/world_of_tanks/ \
    user@linux-server:~/wot_ai/
```

---

## 总结

### WSL 的优势
- ✅ 在 Windows 上开发 Linux 代码
- ✅ 方便的文件系统互访
- ✅ 完整的 Linux 工具链
- ✅ WSL2 支持 GPU（需配置）

### WSL 的限制
- ⚠️ 无法直接访问 Windows GUI
- ⚠️ 跨文件系统性能较差
- ⚠️ 某些系统调用不支持

### 推荐用法
1. **开发**: WSL 中开发和测试 Python 代码
2. **C++ 编译**: Windows 端编译 C++ 模块
3. **游戏客户端**: Windows 端运行
4. **训练服务器**: 真实 Linux 机器上部署

---

## 快速参考

```bash
# WSL 基础命令
wsl --list --verbose          # 查看 WSL 发行版
wsl --set-default Ubuntu      # 设置默认发行版
wsl --shutdown                # 关闭 WSL
wsl --update                  # 更新 WSL

# 访问 Windows 文件
cd /mnt/c/Users/YourName
cd /mnt/d/projects

# 从 Windows 访问 WSL 文件
\\wsl$\Ubuntu\home\username

# 在 WSL 中调用 Windows 程序
cmd.exe /c dir
explorer.exe .
```

---

**推荐阅读**:
- [WSL 官方文档](https://docs.microsoft.com/windows/wsl/)
- [WSL GPU 支持](https://docs.microsoft.com/windows/wsl/tutorials/gpu-compute)
- [最佳实践](https://docs.microsoft.com/windows/wsl/compare-versions)

