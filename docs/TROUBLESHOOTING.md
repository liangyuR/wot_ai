# 故障排查指南

## 常见问题和解决方案

---

## 构建相关

### ❌ 问题 1: build_xmake.bat 报错 ".Content was unexpected"

**症状**:
```
====================================
Building with xmake
====================================
.Content was unexpected at this time.
```

**原因**: xmake 未安装

**解决方案**:

**方式 1: 使用安装脚本（推荐）**
```powershell
# 右键"以管理员身份运行" PowerShell
.\install_xmake.ps1
```

**方式 2: 使用 Scoop**
```powershell
# 先安装 Scoop (如果没有)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# 安装 xmake
scoop install xmake
```

**方式 3: 手动下载**
1. 访问 https://github.com/xmake-io/xmake/releases
2. 下载最新的 `xmake-vX.X.X.win64.exe`
3. 运行安装程序
4. 重启终端

**验证安装**:
```bash
xmake --version
# 应该输出: xmake v2.8.5+...
```

---

### ❌ 问题 2: CMake 找不到 Visual Studio

**症状**:
```
CMake Error: Could not find CMAKE_C_COMPILER
```

**解决方案**:

**选项 A: 安装 Visual Studio**
1. 下载 [Visual Studio 2022 Community](https://visualstudio.microsoft.com/)
2. 安装时选择 "Desktop development with C++"
3. 确保包含 CMake 工具

**选项 B: 改用 xmake（推荐）**
```bash
# xmake 会自动检测编译器
.\install_xmake.ps1
.\build_xmake.bat
```

---

### ❌ 问题 3: C++ 模块编译失败

**症状**:
```
fatal error: pybind11/pybind11.h: No such file or directory
```

**解决方案**:

```bash
# 安装 pybind11
pip install pybind11

# 使用 xmake 自动处理依赖
xmake f -c
xmake
```

---

## 网络连接问题

### ❌ 问题 4: Windows 客户端无法连接 Linux 服务器

**症状**:
```
Failed to connect to server
Connection refused
```

**诊断步骤**:

**1. 检查网络连通性**
```bash
# Windows 上
ping 192.168.1.100

# 如果 ping 不通，检查：
# - IP 地址是否正确
# - 两台机器是否在同一网络
# - 防火墙设置
```

**2. 检查端口**
```bash
# Windows 上测试端口
telnet 192.168.1.100 9999

# 或使用 PowerShell
Test-NetConnection -ComputerName 192.168.1.100 -Port 9999
```

**3. 检查 Linux 服务器**
```bash
# Linux 上检查服务器是否运行
ps aux | grep training_server

# 检查端口监听
sudo netstat -tlnp | grep 9999
# 应该看到: 0.0.0.0:9999

# 检查防火墙
sudo ufw status
sudo ufw allow 9999/tcp
sudo ufw reload
```

**4. 检查配置文件**
```yaml
# configs/client_config.yaml
server:
  host: "192.168.1.100"  # 确认 IP 正确
  port: 9999
```

---

### ❌ 问题 5: 连接延迟过高（>100ms）

**症状**:
- AI 反应迟钝
- 动作执行滞后

**解决方案**:

**1. 降低分辨率和帧率**
```yaml
# configs/client_config.yaml
capture:
  width: 1280    # 降低到 720p
  height: 720
  fps: 20        # 降低帧率

network:
  jpeg_quality: 70  # 降低质量
```

**2. 检查网络质量**
```bash
# Windows 上持续 ping
ping -t 192.168.1.100

# 应该看到延迟 < 10ms
```

**3. 使用有线连接**
- WiFi: 延迟 20-50ms
- 有线: 延迟 1-5ms ✅

---

## GPU 问题

### ❌ 问题 6: GPU 未被使用

**症状**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**解决方案**:

**1. 检查 NVIDIA 驱动**
```bash
# Linux
nvidia-smi

# 应该显示 GPU 信息
```

**2. 检查 CUDA 版本**
```bash
# 检查 CUDA
nvcc --version

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"
```

**3. 重新安装 PyTorch**
```bash
# 卸载
pip uninstall torch torchvision

# 根据 CUDA 版本安装
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**4. 验证**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU count: {torch.cuda.device_count()}")
```

---

### ❌ 问题 7: CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:

**1. 降低 batch size**
```yaml
# configs/ppo_config.yaml
ppo:
  batch_size: 32  # 从 64 降低到 32
```

**2. 清理 GPU 缓存**
```python
import torch
torch.cuda.empty_cache()
```

**3. 检查显存使用**
```bash
watch -n 1 nvidia-smi

# 查看哪个进程占用显存
```

---

## Python 环境问题

### ❌ 问题 8: ModuleNotFoundError

**症状**:
```
ModuleNotFoundError: No module named 'torch'
```

**解决方案**:

**1. 确认虚拟环境已激活**
```bash
# Windows
venv\Scripts\activate

# Linux
source venv/bin/activate

# 应该看到 (venv) 前缀
```

**2. 重新安装依赖**
```bash
pip install -r requirements.txt
```

**3. 检查 Python 版本**
```bash
python --version
# 应该是 Python 3.10 或更高
```

---

### ❌ 问题 9: 导入 cpp_bindings 失败

**症状**:
```python
>>> from cpp_bindings import ScreenCapture
ImportError: DLL load failed
```

**解决方案**:

**1. 重新编译 C++ 模块**
```bash
# 使用 xmake
build_xmake.bat

# 或 CMake
build.bat
```

**2. 检查文件是否存在**
```bash
# Windows
dir python\cpp_bindings.pyd

# Linux
ls python/cpp_bindings.so
```

**3. 使用 Python fallback（临时方案）**
```python
# 系统会自动使用 mss 和 pynput
# 性能略差但功能完整
```

---

## 游戏相关问题

### ❌ 问题 10: AI 不控制坦克

**症状**:
- 客户端连接成功
- 但坦克不移动

**解决方案**:

**1. 确认游戏窗口焦点**
- 游戏窗口必须在前台
- 不能最小化

**2. 检查输入权限**
```bash
# 以管理员身份运行客户端
# 右键 -> 以管理员身份运行
```

**3. 检查防作弊检测**
- 某些反作弊系统可能阻止输入模拟
- 建议在训练场测试

---

### ❌ 问题 11: 屏幕捕获失败

**症状**:
```
Failed to capture screen
Access denied
```

**解决方案**:

**1. 以管理员身份运行**
```bash
# 右键客户端脚本 -> 以管理员身份运行
```

**2. 关闭游戏全屏模式**
- 使用窗口模式或无边框窗口
- 全屏独占模式可能无法捕获

**3. 检查 DPI 缩放**
```bash
# Windows 设置 -> 显示 -> 缩放
# 建议使用 100% 缩放
```

---

## 训练相关问题

### ❌ 问题 12: 训练不收敛

**症状**:
- 训练很久但 reward 没有上升
- AI 行为始终随机

**解决方案**:

**1. 检查奖励函数**
```yaml
# configs/ppo_config.yaml
rewards:
  damage_dealt: 1.0      # 确保奖励合理
  survival_bonus: 0.01   # 不要太小
  death: -20.0           # 惩罚足够大
```

**2. 调整学习率**
```yaml
ppo:
  learning_rate: 1.0e-4  # 尝试更小的学习率
```

**3. 增加训练时间**
- 前 50 万步通常是随机探索
- 200-500 万步才能看到明显效果

**4. 检查 TensorBoard**
```bash
tensorboard --logdir logs/tensorboard
```

查看：
- `rollout/ep_rew_mean` - 应该上升
- `train/loss` - 应该下降
- `train/explained_variance` - 应该接近 1

---

### ❌ 问题 13: 训练中断

**症状**:
- 训练突然停止
- 没有保存检查点

**解决方案**:

**1. 使用检查点恢复**
```bash
# 查找最新的检查点
ls models/checkpoints/

# 恢复训练
python train/train_ppo.py --resume models/checkpoints/wot_ppo_500000_steps.zip
```

**2. 减小保存频率**
```yaml
# configs/ppo_config.yaml
training:
  save_freq: 10000  # 更频繁保存
```

**3. 使用 tmux/screen（Linux）**
```bash
# 创建会话
tmux new -s training

# 运行训练
python train/train_ppo.py

# 分离会话: Ctrl+B, D
# 重新连接: tmux attach -t training
```

---

## 性能问题

### ❌ 问题 14: FPS 过低

**症状**:
- 屏幕捕获 < 30 FPS
- 系统卡顿

**解决方案**:

**1. 降低捕获分辨率**
```yaml
capture:
  width: 1280
  height: 720
```

**2. 使用 C++ 模块**
```bash
# 确保 C++ 模块已编译
build_xmake.bat

# 验证
python -c "from cpp_bindings import ScreenCapture; print('OK')"
```

**3. 关闭不必要的程序**
- 浏览器
- 其他游戏
- 视频播放器

---

### ❌ 问题 15: 训练速度慢

**症状**:
- < 1000 steps/s
- GPU 利用率低

**解决方案**:

**1. 增大 batch size**
```yaml
ppo:
  batch_size: 128  # 利用 GPU
  n_steps: 4096
```

**2. 使用多客户端**
```bash
# 在多台 Windows PC 上运行
python game_client.py --host <linux-ip>
```

**3. 检查瓶颈**
```bash
# Linux 上监控
htop  # CPU 使用率
nvidia-smi  # GPU 使用率
iftop  # 网络使用率
```

---

## 获取更多帮助

### 调试工具

```bash
# 测试安装
python python/tests/test_installation.py

# 测试 C++ 模块
xmake run test_capture
xmake run test_input

# 查看日志
tail -f logs/client.log
tail -f logs/training.log
```

### 日志级别

```yaml
# configs/client_config.yaml
logging:
  level: "DEBUG"  # 获取详细日志
```

### 社区支持

- 📖 [完整文档](../README.md)
- 💬 GitHub Issues
- 📧 技术支持

---

## 预防措施

### 检查清单

在开始之前：
- [ ] Python 3.10+ 已安装
- [ ] GPU 驱动最新
- [ ] CUDA 版本正确
- [ ] 虚拟环境已创建
- [ ] 依赖已安装
- [ ] 防火墙已配置
- [ ] 网络连接正常

### 最佳实践

1. **始终使用虚拟环境**
2. **定期保存检查点**
3. **监控训练进度**
4. **备份重要数据**
5. **阅读文档**

---

**如果问题仍未解决，请提交 GitHub Issue 并附上**：
- 操作系统版本
- Python 版本
- 错误信息完整输出
- 相关配置文件
- 日志文件

