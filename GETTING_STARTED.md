# 🚀 开始使用

## 欢迎！

您已经成功获取了坦克世界 AI 项目的完整代码。本指南将帮助您快速上手。

---

## ⚡ 5 分钟快速决策

### 我应该用哪个方案？

```
┌─────────────────────────────────────────┐
│  您有 Linux 服务器或 GPU 云服务器吗？  │
└───────────┬─────────────────────────────┘
            │
    ┌───────┴───────┐
    │               │
   是              否
    │               │
    ▼               ▼
 方案 A          方案 B
 ⭐⭐⭐⭐⭐      ⭐⭐⭐
 (推荐)         (备选)
```

### 方案 A（推荐）
**Windows 游戏 + Linux 训练服务器**

✅ 最佳性能  
✅ GPU 利用率最高  
✅ 支持多客户端并行  
✅ 可云端部署  

👉 [立即开始](docs/QUICK_DEPLOY_A.md)

### 方案 B（备选）
**纯 Windows 单机部署**

✅ 配置最简单  
⚠️ 性能受限  
⚠️ GPU 被游戏占用  

👉 [单机部署](docs/INSTALLATION.md)

---

## 🎯 方案 A：10 分钟部署

### 准备清单

**Windows PC**：
- [ ] Windows 10/11
- [ ] Python 3.10+
- [ ] Visual Studio（C++ 支持）
- [ ] 《坦克世界》已安装

**Linux 服务器**：
- [ ] Ubuntu 20.04+ / CentOS 8+
- [ ] Python 3.10+
- [ ] NVIDIA GPU (推荐 RTX 3060+)
- [ ] CUDA 12.1+

**网络**：
- [ ] 两台机器可以互相连接
- [ ] 防火墙开放端口 9999

### 第一步：Windows 端

打开 PowerShell：

```powershell
# 1. 进入项目目录
cd d:\projects\world_of_tanks

# 2. 一键构建（自动安装所有依赖）
.\build.bat

# 3. 测试安装
venv\Scripts\activate
python python/tests/test_installation.py
```

**预期输出**：
```
✓ PyTorch: 2.0.0
✓ CUDA available: True
✓ GPU: NVIDIA GeForce RTX 5090
✓ All tests passed!
```

### 第二步：Linux 端

SSH 连接到 Linux 服务器：

```bash
# 1. 创建工作目录
mkdir -p ~/wot_ai
cd ~/wot_ai

# 2. 从 Windows 传输文件
# (在 Windows 上运行)
# scp -r python configs requirements.txt user@<linux-ip>:~/wot_ai/

# 3. 安装环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. 开放防火墙
sudo ufw allow 9999/tcp
```

### 第三步：运行测试

**Linux 服务器** （终端 1）：
```bash
cd ~/wot_ai
source venv/bin/activate
./start_linux_server.sh
```

**Windows PC**：
```powershell
# 1. 启动《坦克世界》
# 2. 进入训练场或战斗
# 3. 运行客户端
cd d:\projects\world_of_tanks
.\start_windows_client.bat
# 输入 Linux 服务器 IP (例如: 192.168.1.100)
```

**成功标志**：
- Linux 显示：`Client connected from...`
- Windows 显示：`Connected to server...`
- AI 开始控制坦克！

### 第四步：开始训练

**Linux 服务器** （终端 2）：
```bash
cd ~/wot_ai
source venv/bin/activate
python python/train/train_ppo.py --config configs/ppo_config.yaml
```

**监控训练** （终端 3）：
```bash
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006
```

浏览器访问：`http://<linux-ip>:6006`

---

## 📖 详细文档

部署成功后，深入学习：

| 文档 | 内容 | 时长 |
|------|------|------|
| [方案 A 快速部署](docs/QUICK_DEPLOY_A.md) | 详细部署步骤 | 10 分钟 |
| [为什么选方案 A](docs/WHY_PLAN_A.md) | 技术对比分析 | 5 分钟 |
| [系统架构](docs/ARCHITECTURE.md) | 深度技术解析 | 20 分钟 |
| [配置说明](configs/ppo_config.yaml) | 参数调优指南 | 随时查阅 |
| [项目总览](PROJECT_OVERVIEW.md) | 完整导航 | 5 分钟 |

---

## ❓ 常见问题

### Q1: C++ 模块编译失败怎么办？

**A**: C++ 模块不是必须的，可以使用 Python fallback：
```python
# 系统会自动回退到 mss 和 pynput
# 性能略有下降但不影响使用
```

如果想解决编译问题：
```bash
# 确保安装了 Visual Studio 2019/2022
# 包含 "Desktop development with C++" 工作负载
```

### Q2: 连接不上 Linux 服务器？

**A**: 检查清单：
```bash
# 1. 测试网络
ping <linux-ip>

# 2. 检查端口
telnet <linux-ip> 9999

# 3. Linux 防火墙
sudo ufw status
sudo netstat -tlnp | grep 9999

# 4. 服务器是否运行
ps aux | grep training_server
```

### Q3: GPU 没有被使用？

**A**: 
```bash
# Linux 上检查
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，重装 PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q4: 延迟太高（>100ms）？

**A**: 优化建议：
```bash
# 降低分辨率和帧率
python game_client.py --width 1280 --height 720 --fps 20

# 或编辑配置文件
# configs/client_config.yaml
network:
  jpeg_quality: 70  # 降低质量
```

### Q5: 训练多久能看到效果？

**A**: 
- **随机探索**: 前 10-50 万步，行为看起来很随机
- **初步学习**: 50-200 万步，开始有简单策略
- **基本能玩**: 200-500 万步，能打中敌人
- **较好水平**: 500-1000 万步，有战术意识
- **高水平**: 1000 万步以上

---

## 🎓 学习路径

### 第 1 天：部署和测试
- [x] 部署 Windows 客户端
- [x] 部署 Linux 服务器
- [x] 运行测试连接
- [x] 观察 AI 行为

### 第 2-3 天：理解系统
- [ ] 阅读系统架构文档
- [ ] 理解 PPO 算法
- [ ] 了解奖励函数设计
- [ ] 观察 TensorBoard

### 第 1 周：开始训练
- [ ] 调整奖励函数
- [ ] 启动正式训练
- [ ] 监控训练进度
- [ ] 测试不同超参数

### 第 2-4 周：优化改进
- [ ] 实现自动瞄准
- [ ] 添加小地图分析
- [ ] 优化动作空间
- [ ] 尝试多客户端训练

---

## 🛠️ 实用命令速查

### 启动系统
```bash
# Linux 服务器
./start_linux_server.sh

# Windows 客户端
.\start_windows_client.bat
```

### 训练模型
```bash
# 从头开始
python python/train/train_ppo.py --config configs/ppo_config.yaml

# 从检查点恢复
python python/train/train_ppo.py --resume models/checkpoints/xxx.zip
```

### 监控训练
```bash
# TensorBoard
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006

# 查看 GPU 使用
watch -n 1 nvidia-smi
```

### 测试模型
```bash
# 使用训练好的模型
python python/network/training_server.py \
    --model models/best/best_model.zip \
    --host 0.0.0.0 --port 9999
```

### 同步代码
```bash
# Windows → Linux
.\scripts\sync_to_linux.bat

# 或手动
scp -r python user@linux-ip:~/wot_ai/
```

---

## 📊 性能优化建议

### 网络优化

**局域网（最佳）**：
```yaml
# configs/client_config.yaml
capture:
  fps: 30
  width: 1920
  height: 1080
network:
  jpeg_quality: 80
```

**远程/云端**：
```yaml
capture:
  fps: 20
  width: 1280
  height: 720
network:
  jpeg_quality: 70
```

### GPU 优化

**RTX 5090 推荐配置**：
```yaml
# configs/ppo_config.yaml
ppo:
  batch_size: 128  # 大 batch
  n_steps: 4096    # 更多步数

training:
  device: "cuda"
  n_envs: 1        # 单客户端
```

**多客户端并行**（需要多台 Windows PC）：
```yaml
training:
  n_envs: 4        # 4 个并行环境
```

---

## 🎯 里程碑目标

### 短期（本周）
- [x] 成功部署系统
- [ ] 运行 10 万步训练
- [ ] 理解系统架构
- [ ] 观察学习曲线

### 中期（本月）
- [ ] 完成 100 万步训练
- [ ] 优化奖励函数
- [ ] 实现基本策略
- [ ] 测试不同配置

### 长期（3 个月）
- [ ] 完成 1000 万步训练
- [ ] 实现高级功能（自动瞄准、路径规划）
- [ ] 多客户端并行训练
- [ ] 云端部署实验

---

## 🌟 成功案例

### 训练效果示例

**0-50 万步**：
```
行为: 随机移动，乱开炮
存活: <30 秒
伤害: ~100
```

**50-200 万步**：
```
行为: 学会躲避，开始瞄准
存活: 1-2 分钟
伤害: ~500
```

**500 万步+**：
```
行为: 战术移动，精准射击
存活: 3-5 分钟
伤害: 1500+
胜率: 开始上升
```

---

## 📞 获取帮助

### 文档
- 📖 [完整 README](README.md)
- 🏗️ [系统架构](docs/ARCHITECTURE.md)
- 🚀 [快速部署](docs/QUICK_DEPLOY_A.md)
- 🔍 [项目总览](PROJECT_OVERVIEW.md)

### 遇到问题？
1. 查看 [故障排查](docs/DISTRIBUTED_SETUP.md#故障排查)
2. 搜索 GitHub Issues
3. 提交新 Issue

### 想贡献？
- 🐛 报告 Bug
- ✨ 提议新功能
- 📝 改进文档
- 💻 提交代码

---

## ⚠️ 重要提醒

1. **法律警告**：使用 AI 可能违反游戏条款，仅供学习研究
2. **账号安全**：建议使用测试账号
3. **数据隐私**：不收集个人信息
4. **性能影响**：训练时 GPU 会满载运行

---

## 🎉 准备好了吗？

选择您的起点：

**🚀 我要马上开始**
→ [10 分钟快速部署](docs/QUICK_DEPLOY_A.md)

**📚 我想先了解原理**
→ [系统架构详解](docs/ARCHITECTURE.md)

**💡 我想知道为什么选方案 A**
→ [方案对比分析](docs/WHY_PLAN_A.md)

**🗺️ 我需要完整地图**
→ [项目总览](PROJECT_OVERVIEW.md)

---

**祝您在 AI 坦克世界的旅程中收获满满！** 🎮🤖

