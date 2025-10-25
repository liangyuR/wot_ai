# 快速开始指南

## ⚠️ 警告

**使用本软件可能违反《坦克世界》服务条款，导致账号被封禁。仅用于学习和研究！**

## 工作流程概览

```
收集数据 → 训练模型 → 测试模型 → 优化调整
```

## 步骤 1: 收集训练数据（可选）

如果您想使用模仿学习，首先需要收集游戏数据：

```bash
# 激活虚拟环境
venv\Scripts\activate

# 运行数据收集工具
python python/data_collection/record_gameplay.py --fps 30
```

**操作流程**:
1. 运行脚本后，切换到《坦克世界》
2. 按任意键开始录制
3. 正常游戏（尽量展示好的操作）
4. 按 ESC 停止录制
5. 数据保存在 `data/recordings/` 目录

**建议**:
- 录制至少 10-20 场对局
- 覆盖不同地图和坦克类型
- 展示多样化的战术（进攻、防守、侦察等）

## 步骤 2: 配置训练参数

编辑 `configs/ppo_config.yaml`：

```yaml
# 关键参数
training:
  total_timesteps: 10000000  # 训练步数（可根据GPU性能调整）
  
env:
  screen:
    width: 1920  # 游戏分辨率
    height: 1080
    
ppo:
  learning_rate: 3.0e-4  # 学习率
  n_steps: 2048  # 每次更新的步数
```

## 步骤 3: 开始训练

```bash
# 从头开始训练
python python/train/train_ppo.py --config configs/ppo_config.yaml

# 从检查点恢复训练
python python/train/train_ppo.py --resume models/checkpoints/wot_ppo_50000_steps.zip
```

**训练前准备**:
1. 启动《坦克世界》
2. 进入训练场或随机战斗
3. 确保游戏窗口可见（不要最小化）
4. 运行训练脚本

**训练时长估算**（RTX 5090）:
- 100万步：约 2-4 小时
- 1000万步：约 20-40 小时

**监控训练**:
```bash
# 在另一个终端启动 TensorBoard
tensorboard --logdir logs/tensorboard
```

访问 http://localhost:6006 查看训练进度。

## 步骤 4: 测试训练好的模型

```bash
# 运行 AI Bot
python python/run_bot.py --model models/best/best_model.zip --episodes 5
```

**测试流程**:
1. 启动《坦克世界》并进入战斗
2. 运行测试脚本
3. AI 将自动控制坦克
4. 观察表现并记录数据

## 步骤 5: 评估和优化

### 查看性能指标

```python
# 评估脚本
python python/eval/evaluate_model.py --model models/best/best_model.zip
```

**关键指标**:
- 平均存活时间
- 平均伤害输出
- 击杀/死亡比
- 胜率

### 调优建议

**如果 AI 表现不佳**:

1. **增加训练步数**: 提高 `total_timesteps`
2. **调整奖励函数**: 修改 `configs/ppo_config.yaml` 中的 `rewards`
3. **改变学习率**: 降低 `learning_rate`（如 1e-4）
4. **使用课程学习**: 启用 `curriculum.enabled: true`

**如果训练不稳定**:
- 降低 `learning_rate`
- 减小 `clip_range`
- 增加 `n_steps`

## 步骤 6: 可视化和调试

### 可视化检测结果

```python
from vision.detector import TankDetector
import cv2

detector = TankDetector()
# 测试检测...
```

### 查看环境状态

```bash
# 运行环境测试
python python/tests/test_env.py
```

## 常见使用场景

### 场景 1: 快速测试（无训练）

使用预训练的随机策略测试环境：

```bash
python python/run_bot.py --model random --episodes 1
```

### 场景 2: 短时间训练测试

快速验证训练管道（10分钟）：

```bash
python python/train/train_ppo.py --timesteps 10000
```

### 场景 3: 完整训练流程

```bash
# 1. 收集数据（1小时）
python python/data_collection/record_gameplay.py

# 2. 长时间训练（一夜）
python python/train/train_ppo.py --timesteps 10000000

# 3. 评估模型（30分钟）
python python/eval/evaluate_model.py --model models/best/best_model.zip --episodes 20
```

## 性能优化提示

### GPU 利用率

确保 GPU 被充分利用：

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 并行训练（实验性）

```yaml
# configs/ppo_config.yaml
training:
  n_envs: 4  # 并行4个环境（需要多个游戏实例）
```

## 下一步

- [训练指南](TRAINING.md) - 深入了解训练细节
- [配置说明](CONFIGURATION.md) - 所有配置参数详解
- [API 文档](API.md) - 代码接口文档
- [故障排查](TROUBLESHOOTING.md) - 常见问题解决

## 获取帮助

遇到问题？
1. 查看 [故障排查文档](TROUBLESHOOTING.md)
2. 检查 GitHub Issues
3. 阅读代码注释和文档字符串

