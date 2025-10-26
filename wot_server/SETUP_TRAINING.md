# 训练环境搭建指南

## 快速开始

### 1. 安装依赖

```bash
cd wot_server

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 验证安装

```bash
# 测试数据加载器
python data/action_mapper.py
python data/dataset.py

# 测试环境
python env/wot_env.py
```

### 3. 开始训练

**方法A：使用现有录制数据训练**

```bash
# Linux
bash train_imitation.sh

# Windows
train_imitation.bat
```

**方法B：自定义参数**

```bash
python train/train_imitation.py \
    --data-dir ../wot_client/data/recordings \
    --epochs 100 \
    --batch-size 64
```

## 依赖项

核心依赖（`requirements.txt`）：

```
# 深度学习
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.12.0

# 强化学习
gymnasium>=0.28.0
stable-baselines3>=2.0.0

# 计算机视觉
opencv-python>=4.7.0
numpy>=1.24.0

# 工具
pyyaml>=6.0
loguru>=0.7.0
tqdm>=4.65.0
```

安装GPU支持（可选）：

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 目录结构

```
wot_server/
├── agents/              # PPO智能体
├── configs/             # 配置文件
├── data/                # 数据加载模块 [新增]
│   ├── action_mapper.py    # 动作映射器
│   └── dataset.py          # 数据集加载器
├── env/                 # 环境模块 [新增]
│   └── wot_env.py          # 离线训练环境
├── train/               # 训练脚本
│   ├── train_ppo.py        # PPO训练（在线）
│   └── train_imitation.py  # 行为克隆训练（离线）[新增]
├── models/              # 模型保存目录
│   ├── checkpoints/
│   └── best/
└── logs/                # 日志目录
    ├── tensorboard/
    └── imitation/
```

## 训练流程

### 阶段1：数据采集（wot_client）

在Windows客户端录制游戏数据：

```bash
cd wot_client
python record_main.py
```

**目标**：
- 至少10场战斗
- 每场5-10分钟
- 总共30,000+帧

### 阶段2：行为克隆训练（wot_server）

使用录制数据训练初始模型：

```bash
cd wot_server
bash train_imitation.sh
```

**输出**：
- `models/checkpoints/best_imitation_model.pth`

### 阶段3：在线强化学习（可选）

使用PPO继续优化：

```bash
bash start_training.sh
```

### 阶段4：部署测试

让AI实际玩游戏：

```bash
cd wot_client
python client_main.py --model ../wot_server/models/checkpoints/best_imitation_model.pth
```

## 配置说明

编辑 `configs/ppo_config.yaml`：

```yaml
# 数据路径
data:
  recordings_dir: "../wot_client/data/recordings"

# 训练参数
imitation:
  learning_rate: 1.0e-4
  batch_size: 32
  epochs: 50

# 硬件
training:
  device: "cuda"  # 或 "cpu"
```

## 常见问题

### Q: Dataset is empty

**A:** 检查数据路径是否正确：

```bash
ls ../wot_client/data/recordings/
# 应该看到 session_YYYYMMDD_HHMMSS/ 目录
```

### Q: CUDA out of memory

**A:** 减小batch_size或使用CPU：

```bash
python train/train_imitation.py --batch-size 16
```

或修改配置：

```yaml
training:
  device: "cpu"
```

### Q: 训练很慢

**A:** 
- 使用GPU加速
- 增加`num_workers`
- 减小图像尺寸

### Q: 准确率低于0.5

**A:**
- 录制更多数据
- 提高数据质量（避免错误操作）
- 增加训练轮数
- 调整学习率

## 监控训练

使用TensorBoard：

```bash
tensorboard --logdir=./logs/imitation
```

访问：http://localhost:6006

关键指标：
- **Loss**: 应该持续下降
- **Accuracy**: 应该 > 0.7
- **Train vs Val**: 差距不应太大（避免过拟合）

## 下一步

1. ✅ **采集数据** - 录制10+场战斗
2. ✅ **训练模型** - 运行行为克隆
3. ⏳ **评估模型** - 检查准确率
4. ⏳ **部署测试** - 让AI玩游戏
5. ⏳ **在线优化** - 使用PPO微调

## 获取帮助

- 详细文档：`README_IMITATION.md`
- 问题追踪：GitHub Issues
- 示例配置：`configs/ppo_config.yaml`

