# 行为克隆训练指南

## 概述

行为克隆（Behavioral Cloning）是一种监督学习方法，通过让AI模仿人类专家的操作来学习游戏策略。

## 工作流程

```
1. 录制数据 (wot_client)
   ↓
2. 训练模型 (wot_server)
   ↓
3. 部署测试 (game_client.py)
```

## 快速开始

### 1. 录制游戏数据

在 Windows 客户端：

```bash
cd wot_client
python record_main.py
# 或者直接运行
start_recording.bat
```

- 按 **F9** 开始录制
- 正常游戏 5-10 分钟
- 按 **F10** 停止录制
- 重复录制多场战斗（建议 10+ 场）

数据保存在：`wot_client/data/recordings/session_YYYYMMDD_HHMMSS/`

### 2. 训练模型

在 Linux/Windows 服务器：

```bash
cd wot_server

# Linux
bash train_imitation.sh

# Windows
train_imitation.bat
```

或使用命令行参数：

```bash
python train/train_imitation.py \
    --config configs/ppo_config.yaml \
    --data-dir ../wot_client/data/recordings \
    --epochs 50 \
    --batch-size 32 \
    --val-split 0.2
```

### 3. 监控训练进度

使用 TensorBoard：

```bash
tensorboard --logdir=./logs/imitation
```

在浏览器打开：http://localhost:6006

## 配置说明

编辑 `configs/ppo_config.yaml`：

```yaml
# 数据配置
data:
  recordings_dir: "../wot_client/data/recordings"
  frame_skip: 1  # 采样间隔
  augment: true  # 数据增强

# 训练配置
imitation:
  learning_rate: 1.0e-4
  batch_size: 32
  epochs: 50
  val_split: 0.2
  image_size: [256, 256]
```

## 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `configs/ppo_config.yaml` |
| `--data-dir` | 录制数据目录 | `../wot_client/data/recordings` |
| `--epochs` | 训练轮数 | 50 |
| `--batch-size` | 批量大小 | 32 |
| `--val-split` | 验证集比例 | 0.2 |
| `--resume` | 恢复训练的检查点 | None |

## 数据要求

### 最小数据量
- **测试**：100+ 帧（约 5 秒）
- **基础训练**：3,000+ 帧（约 3 分钟）
- **实用模型**：30,000+ 帧（约 30 分钟）
- **高质量模型**：100,000+ 帧（约 2 小时）

### 数据质量
- ✅ 多样化场景（不同地图、坦克）
- ✅ 高质量操作（熟练玩家录制）
- ✅ 平衡的动作分布
- ❌ 避免过多停顿/等待
- ❌ 避免错误操作

## 输出文件

训练完成后生成：

```
models/
├── checkpoints/
│   ├── best_imitation_model.pth      # 最佳模型
│   ├── final_imitation_model.pth     # 最终模型
│   └── checkpoint_epoch_N.pth        # 定期检查点
│
logs/
└── imitation/
    └── events.out.tfevents.*          # TensorBoard日志
```

## 评估模型

检查训练日志：

```
Epoch 50/50
  Train - Loss: 0.3245, Acc: 0.8567
  Val   - Loss: 0.3891, Acc: 0.8234
```

**好的指标：**
- Validation Accuracy > 0.7
- Train/Val Loss 接近（没有过拟合）

**需要改进：**
- Accuracy < 0.5：数据量不足或质量差
- Val Loss >> Train Loss：过拟合，需要更多数据

## 故障排查

### 1. Dataset is empty

**原因**：数据目录不存在或没有录制数据

**解决**：
```bash
# 检查数据目录
ls ../wot_client/data/recordings/

# 确保有 session 文件夹
```

### 2. CUDA out of memory

**原因**：GPU内存不足

**解决**：
- 减小 `batch_size`（如 32 → 16）
- 减小 `image_size`（如 256 → 128）
- 使用CPU训练（慢但稳定）

```bash
# 修改配置文件
training:
  device: "cpu"
```

### 3. 训练过慢

**加速方法**：
- 使用GPU（CUDA）
- 增加 `num_workers`（数据加载线程）
- 减小图像分辨率
- 增加 `frame_skip`（跳过部分帧）

### 4. 准确率不提升

**可能原因**：
- 数据量不足 → 录制更多
- 数据质量差 → 重新录制高质量数据
- 学习率不合适 → 调整 `learning_rate`
- 模型欠拟合 → 增加训练轮数

## 下一步

训练完成后：

1. **转换模型**：将PyTorch模型转换为PPO格式（如需）
2. **部署测试**：使用 `game_client.py` 连接到游戏
3. **在线微调**：使用PPO在真实游戏中继续训练

## 进阶话题

### 数据增强

自动启用的增强：
- 随机亮度调整
- 随机对比度
- 随机噪声

### 不平衡数据处理

如果某些动作过多（如"停止"），可以：
- 使用类别权重
- 过采样少数类
- 欠采样多数类

### 序列建模

对于需要记忆的任务，使用：
```python
from data.dataset import WotSequenceDataset

seq_dataset = WotSequenceDataset(
    base_dataset=dataset,
    sequence_length=4
)
```

## 参考资料

- [Behavioral Cloning in Autonomous Driving](https://arxiv.org/abs/1604.07316)
- [DAgger: Dataset Aggregation](https://arxiv.org/abs/1011.0686)
- [Imitation Learning Tutorial](https://sites.google.com/view/icml2018-imitation-learning/)

