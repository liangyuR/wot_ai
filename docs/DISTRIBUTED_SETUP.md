# 分布式部署指南（方案A）

## 架构概览

```
┌─────────────────────────────────┐         ┌──────────────────────────────────┐
│     Windows PC (游戏端)          │         │   Linux Server (训练端)          │
│                                  │         │                                  │
│  ┌────────────────────────────┐ │         │  ┌────────────────────────────┐  │
│  │  World of Tanks 客户端     │ │         │  │  PyTorch + GPU 训练        │  │
│  └────────────────────────────┘ │         │  └────────────────────────────┘  │
│              │                   │         │              ▲                   │
│              ▼                   │         │              │                   │
│  ┌────────────────────────────┐ │  TCP/IP │  ┌────────────────────────────┐  │
│  │  Game Client               │ │◄───────►│  │  Training Server           │  │
│  │  - 屏幕捕获 (C++)          │ │  9999   │  │  - 模型推理                │  │
│  │  - 输入控制                │ │         │  │  - 数据收集                │  │
│  │  - 网络传输                │ │         │  │  - 模型训练                │  │
│  └────────────────────────────┘ │         │  └────────────────────────────┘  │
│                                  │         │                                  │
│  硬件要求:                       │         │  硬件要求:                       │
│  - Windows 10/11                │         │  - Ubuntu 20.04+                │
│  - 8GB RAM                      │         │  - 32GB+ RAM                    │
│  - 稳定网络                     │         │  - NVIDIA RTX 5090              │
│                                  │         │  - 100GB+ 存储                  │
└─────────────────────────────────┘         └──────────────────────────────────┘
```

## 优势分析

### ✅ Windows 端
- 游戏原生运行，性能最优
- 屏幕捕获和输入控制简单可靠
- 无需担心兼容性问题

### ✅ Linux 端
- 更好的 AI/ML 生态系统
- 稳定的长时间训练
- 更高效的资源利用
- 便于云端部署

### ✅ 整体架构
- **职责分离**：游戏交互和模型训练解耦
- **资源优化**：Windows PC 低配置，Linux 服务器高性能
- **可扩展**：可以多个 Windows 客户端连接同一训练服务器
- **灵活性**：训练服务器可以在本地或云端

## 第一步：Windows 端配置

### 1.1 安装依赖

```bash
# 在 Windows PC 上
cd d:\projects\world_of_tanks
build.bat
```

### 1.2 测试屏幕捕获

```bash
venv\Scripts\activate
python python/tests/test_installation.py
```

### 1.3 配置网络客户端

编辑 `configs/client_config.yaml`：

```yaml
server:
  host: "192.168.1.100"  # Linux 服务器 IP
  port: 9999
  
capture:
  width: 1920
  height: 1080
  fps: 30
```

## 第二步：Linux 端配置

### 2.1 创建工作目录

```bash
# 在 Linux 服务器上
mkdir -p ~/wot_ai
cd ~/wot_ai
```

### 2.2 传输项目文件

**方法 A：从 Windows 传输**

```bash
# 在 Windows PowerShell 中
scp -r d:\projects\world_of_tanks\python user@linux-server:~/wot_ai/
scp -r d:\projects\world_of_tanks\configs user@linux-server:~/wot_ai/
scp d:\projects\world_of_tanks\requirements.txt user@linux-server:~/wot_ai/
```

**方法 B：Git 克隆**

```bash
# 在 Linux 服务器上
cd ~/wot_ai
git clone <your-repo-url> .
```

### 2.3 安装 Python 环境

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.10 python3-pip python3-venv

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 验证 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2.4 配置防火墙

```bash
# 开放端口 9999
sudo ufw allow 9999/tcp
sudo ufw reload

# 或者使用 iptables
sudo iptables -A INPUT -p tcp --dport 9999 -j ACCEPT
```

## 第三步：运行分布式系统

### 3.1 启动 Linux 训练服务器

```bash
# 在 Linux 服务器上
cd ~/wot_ai
source venv/bin/activate

# 使用随机策略（测试）
python python/network/training_server.py \
    --host 0.0.0.0 \
    --port 9999 \
    --config configs/ppo_config.yaml

# 或使用已训练的模型
python python/network/training_server.py \
    --host 0.0.0.0 \
    --port 9999 \
    --model models/best_model.zip \
    --config configs/ppo_config.yaml
```

**输出应该显示**：
```
Server listening on 0.0.0.0:9999
Waiting for game client to connect...
```

### 3.2 启动 Windows 游戏客户端

```bash
# 在 Windows PC 上
cd d:\projects\world_of_tanks
venv\Scripts\activate

# 连接到 Linux 服务器
python python/network/game_client.py \
    --host 192.168.1.100 \
    --port 9999 \
    --fps 30
```

**输出应该显示**：
```
Connected to server 192.168.1.100:9999
Using C++ screen capture
Using C++ input control
Starting game loop at 30 FPS
```

### 3.3 启动游戏

1. 启动《坦克世界》
2. 进入训练场或随机战斗
3. AI 将自动控制坦克

## 第四步：数据收集和训练

### 4.1 收集游戏数据

```bash
# Linux 服务器端（保存数据）
python python/network/training_server.py \
    --host 0.0.0.0 \
    --port 9999 \
    --save-data data/collected
```

游戏几局后，停止服务器（Ctrl+C），数据将保存到 `data/collected/`。

### 4.2 开始训练

```bash
# 在 Linux 服务器上
cd ~/wot_ai
source venv/bin/activate

# 训练模型
python python/train/train_ppo.py \
    --config configs/ppo_config.yaml \
    --timesteps 10000000

# 监控训练（另一个终端）
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006
```

在浏览器访问：`http://<linux-server-ip>:6006`

### 4.3 使用训练好的模型

```bash
# 重启训练服务器，加载新模型
python python/network/training_server.py \
    --host 0.0.0.0 \
    --port 9999 \
    --model models/best/best_model.zip
```

## 高级配置

### 多客户端支持

修改 `training_server.py` 支持多个 Windows 客户端：

```python
# 启动多客户端模式
python python/network/training_server.py \
    --host 0.0.0.0 \
    --port 9999 \
    --multi-client \
    --max-clients 4
```

### 云端部署

#### AWS EC2

```bash
# 启动 p4d.24xlarge (8x A100)
aws ec2 run-instances \
    --image-id ami-xxxxxxxxx \
    --instance-type p4d.24xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxxxxxx

# 获取公网 IP
aws ec2 describe-instances --instance-ids i-xxxxxxxxx

# Windows 客户端连接
python python/network/game_client.py \
    --host <ec2-public-ip> \
    --port 9999
```

#### Google Cloud

```bash
# 启动 a2-highgpu-8g (8x A100)
gcloud compute instances create wot-training \
    --machine-type=a2-highgpu-8g \
    --zone=us-central1-a \
    --image-family=pytorch-latest-gpu

# 开放端口
gcloud compute firewall-rules create wot-ai-port \
    --allow tcp:9999
```

### 性能优化

#### 网络优化

```yaml
# configs/client_config.yaml
network:
  compression: true        # JPEG 压缩
  quality: 80             # 压缩质量
  buffer_size: 8192       # 缓冲区大小
  timeout: 5.0            # 超时时间
```

#### 帧率自适应

```python
# 根据网络延迟自动调整 FPS
python python/network/game_client.py \
    --host 192.168.1.100 \
    --port 9999 \
    --adaptive-fps
```

## 故障排查

### 问题 1：无法连接服务器

**检查清单**：
```bash
# Linux 端
# 1. 检查服务器是否运行
ps aux | grep training_server

# 2. 检查端口是否监听
netstat -tlnp | grep 9999

# 3. 检查防火墙
sudo ufw status

# Windows 端
# 4. 测试网络连通性
ping 192.168.1.100
telnet 192.168.1.100 9999
```

### 问题 2：延迟过高

**解决方案**：
```bash
# 降低传输分辨率
python python/network/game_client.py \
    --width 1280 \
    --height 720 \
    --fps 20
```

### 问题 3：GPU 未使用

```bash
# Linux 端检查
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 确保 CUDA 版本匹配
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 性能基准

### 网络带宽需求

| 分辨率 | FPS | 压缩质量 | 带宽 |
|--------|-----|----------|------|
| 1920x1080 | 30 | 80% | ~15 Mbps |
| 1280x720 | 30 | 80% | ~8 Mbps |
| 1920x1080 | 20 | 70% | ~8 Mbps |

### 训练性能（RTX 5090）

| 配置 | 吞吐量 | 预计时间 |
|------|--------|----------|
| 单客户端 | ~5000 steps/s | 1000万步 ≈ 30h |
| 4客户端并行 | ~18000 steps/s | 1000万步 ≈ 8h |

## 下一步

- [配置说明](CONFIGURATION.md) - 详细配置参数
- [训练技巧](TRAINING.md) - 训练优化建议
- [API 文档](API.md) - 网络协议说明

## 总结

方案 A 的优势：
- ✅ **最佳兼容性**：游戏在 Windows 原生运行
- ✅ **性能优化**：Linux 服务器专注训练
- ✅ **灵活部署**：支持本地和云端
- ✅ **易于调试**：前后端独立测试
- ✅ **可扩展性**：支持多客户端并行训练

