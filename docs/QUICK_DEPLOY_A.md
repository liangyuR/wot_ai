# 方案 A 快速部署指南

## 10 分钟部署完整系统

本指南将帮助您快速部署 **Windows 游戏 + Linux 训练服务器** 架构。

---

## 前置要求

### Windows PC
- ✅ Windows 10/11 (64位)
- ✅ Python 3.10+
- ✅ Visual Studio 2019+ (C++ 支持)
- ✅ 《坦克世界》客户端
- ✅ 8GB+ RAM
- ✅ 网络连接

### Linux Server
- ✅ Ubuntu 20.04+ / CentOS 8+
- ✅ Python 3.10+
- ✅ NVIDIA RTX 5090 (或其他 GPU)
- ✅ CUDA 12.1+
- ✅ 32GB+ RAM
- ✅ 100GB+ 存储

---

## 第一步：Windows 端配置（5 分钟）

### 1.1 克隆项目

```bash
cd d:\projects
git clone <your-repo> world_of_tanks
cd world_of_tanks
```

### 1.2 一键构建

```bash
# 运行构建脚本（会自动安装所有依赖和编译 C++ 模块）
build.bat
```

**等待完成后，应该看到**：
```
====================================
Build completed successfully!
====================================
```

### 1.3 测试安装

```bash
venv\Scripts\activate
python python/tests/test_installation.py
```

**所有测试应该通过（C++ bindings 可选）**

---

## 第二步：Linux 端配置（5 分钟）

### 2.1 创建工作目录

```bash
mkdir -p ~/wot_ai
cd ~/wot_ai
```

### 2.2 从 Windows 传输文件

**在 Windows PowerShell 中运行**：

```powershell
# 方式 1: 使用同步脚本（推荐）
cd d:\projects\world_of_tanks
.\scripts\sync_to_linux.sh

# 方式 2: 手动 SCP
scp -r python user@<linux-ip>:~/wot_ai/
scp -r configs user@<linux-ip>:~/wot_ai/
scp requirements.txt user@<linux-ip>:~/wot_ai/
```

### 2.3 安装 Python 环境

```bash
# 在 Linux 服务器上
cd ~/wot_ai

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.4 验证 GPU

```bash
# 检查 CUDA
nvidia-smi

# 检查 PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**应该看到**：
```
CUDA: True
GPU: NVIDIA GeForce RTX 5090
```

### 2.5 开放防火墙

```bash
# Ubuntu/Debian
sudo ufw allow 9999/tcp
sudo ufw reload

# CentOS/RHEL
sudo firewall-cmd --add-port=9999/tcp --permanent
sudo firewall-cmd --reload
```

---

## 第三步：运行系统（测试）

### 3.1 启动 Linux 训练服务器

```bash
# 在 Linux 服务器上
cd ~/wot_ai
source venv/bin/activate

# 快速启动（推荐）
./start_linux_server.sh

# 或手动启动
python python/network/training_server.py \
    --host 0.0.0.0 \
    --port 9999 \
    --config configs/ppo_config.yaml
```

**应该看到**：
```
====================================
World of Tanks AI - Training Server
====================================
Server listening on 0.0.0.0:9999
Waiting for game client to connect...
```

### 3.2 启动 Windows 客户端

**在 Windows PC 上**：

1. 启动《坦克世界》
2. 进入训练场或随机战斗

然后运行：

```bash
cd d:\projects\world_of_tanks
venv\Scripts\activate

# 快速启动（推荐）
start_windows_client.bat

# 或手动启动
python python/network/game_client.py ^
    --host <linux-server-ip> ^
    --port 9999 ^
    --fps 30
```

**输入 Linux 服务器 IP**（例如：192.168.1.100）

**应该看到**：
```
Connected to server 192.168.1.100:9999
Using C++ screen capture
Using C++ input control
Starting game loop at 30 FPS
```

### 3.3 观察运行

**Linux 服务器端**：
```
Client connected from ('192.168.1.XXX', 12345)
Starting inference loop...
Processed 100 frames
Processed 200 frames
...
```

**Windows 客户端**：
- AI 开始控制坦克
- 屏幕每秒捕获 30 帧
- 动作自动执行

---

## 第四步：训练模型

### 4.1 收集数据（可选）

如果想先收集数据再训练：

```bash
# Linux 服务器端
python python/network/training_server.py \
    --host 0.0.0.0 \
    --port 9999 \
    --save-data data/collected_$(date +%Y%m%d)
```

玩几局后停止（Ctrl+C），数据保存到 `data/collected_xxx/`

### 4.2 开始训练

**在 Linux 服务器上（新终端）**：

```bash
cd ~/wot_ai
source venv/bin/activate

# 启动训练
python python/train/train_ppo.py \
    --config configs/ppo_config.yaml \
    --timesteps 10000000
```

**预计时间**：
- RTX 5090: 20-30 小时
- 多客户端并行: 8-12 小时

### 4.3 监控训练

**在 Linux 服务器上（新终端）**：

```bash
cd ~/wot_ai
source venv/bin/activate
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006
```

**在浏览器访问**：`http://<linux-server-ip>:6006`

---

## 第五步：使用训练好的模型

### 5.1 停止当前服务器

在 Linux 服务器上按 `Ctrl+C` 停止当前的训练服务器。

### 5.2 加载新模型

```bash
python python/network/training_server.py \
    --host 0.0.0.0 \
    --port 9999 \
    --model models/best/best_model.zip \
    --config configs/ppo_config.yaml
```

### 5.3 重新连接 Windows 客户端

Windows 端会自动重连，或者重新运行：

```bash
start_windows_client.bat
```

现在 AI 使用的是训练好的模型！

---

## 常见问题速查

### ❌ Windows 客户端无法连接

**检查**：
```bash
# 1. 测试网络连通性
ping <linux-server-ip>

# 2. 测试端口
telnet <linux-server-ip> 9999

# 3. 检查 Linux 防火墙
# 在 Linux 上运行
sudo ufw status
sudo netstat -tlnp | grep 9999
```

### ❌ GPU 未被使用

**检查**：
```bash
# Linux 上
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，重新安装 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### ❌ 延迟太高

**优化**：
```bash
# 降低分辨率和帧率
python python/network/game_client.py \
    --host <linux-ip> \
    --port 9999 \
    --width 1280 \
    --height 720 \
    --fps 20
```

### ❌ C++ 模块编译失败

**解决**：
```bash
# Windows 上
# 1. 确保安装了 Visual Studio 2019/2022
# 2. 安装 CMake
# 3. 重新运行
build.bat

# 如果仍然失败，可以使用 Python fallback
# C++ 模块不是必须的，只是性能会略差
```

---

## 性能优化建议

### 网络优化

**局域网（推荐）**：
- 延迟: 30-50ms
- 带宽: 15-20 Mbps
- 配置: 默认即可

**远程（云端）**：
```yaml
# configs/client_config.yaml
network:
  jpeg_quality: 70  # 降低质量减少带宽
  
capture:
  fps: 20  # 降低帧率
  width: 1280
  height: 720
```

### GPU 利用率优化

```yaml
# configs/ppo_config.yaml
training:
  batch_size: 128  # 5090 可以用更大的 batch
  n_envs: 4        # 多个并行环境（需要多个游戏客户端）
```

### 多客户端并行训练

**启动多个 Windows 客户端**（如果有多台 PC）：

```bash
# PC 1
python python/network/game_client.py --host <linux-ip> --port 9999

# PC 2
python python/network/game_client.py --host <linux-ip> --port 9999

# PC 3
python python/network/game_client.py --host <linux-ip> --port 9999
```

训练速度提升 3-4 倍！

---

## 架构图总结

```
[Windows PC]              [Linux Server]
    │                          │
    │  1. 捕获屏幕 (60fps)     │
    │  ────────────────────►   │  2. AI 推理 (<10ms)
    │                          │
    │  3. 发送画面 (JPEG)      │
    │  ────────────────────►   │  4. 训练模型
    │                          │
    │  5. 接收动作              │
    │  ◄────────────────────   │  6. 返回动作
    │                          │
    │  7. 执行动作              │
    │                          │
```

**总延迟**: 30-50ms (局域网)

---

## 下一步

✅ **系统运行成功！现在您可以**：

1. 📊 [监控训练进度](http://linux-ip:6006) - TensorBoard
2. 📈 [优化超参数](CONFIGURATION.md) - 调整配置
3. 🎯 [改进奖励函数](TRAINING.md) - 训练技巧
4. 🔧 [扩展功能](ARCHITECTURE.md) - 添加新特性
5. ☁️ [云端部署](DISTRIBUTED_SETUP.md#云端部署) - AWS/GCP

---

## 技术支持

- 📖 [完整文档](../README.md)
- 🏗️ [架构详解](ARCHITECTURE.md)
- 🐛 [故障排查](DISTRIBUTED_SETUP.md#故障排查)
- 💬 GitHub Issues

---

**恭喜！您已经成功部署了一个分布式的坦克世界 AI 系统！** 🎉

