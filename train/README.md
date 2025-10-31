# 坦克世界模仿学习训练

基于收集的游戏数据进行端到端行为克隆训练。

## 文件说明

- `train_imitation.py` - 主训练脚本
- `test_model.py` - 模型测试脚本  
- `run_training.py` - 训练启动脚本
- `start_training.bat` - Windows批处理启动脚本
- `requirements.txt` - Python依赖包

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动训练

**Windows:**
```bash
start_training.bat
```

**Linux/WSL:**
```bash
python run_training.py
```

### 3. 测试模型

```bash
# 测试单张图像
python test_model.py --image path/to/frame.jpg

# 在数据集上测试
python test_model.py --data_root data_collection/data/recordings --num_samples 20
```

## 数据格式

训练脚本会自动读取 `data_collection/data/recordings/` 下的所有session数据：

```
data_collection/data/recordings/
├── session_20251030_010735/
│   ├── actions.json
│   └── frames/
│       ├── frame_0000.jpg
│       ├── frame_0001.jpg
│       └── ...
├── session_20251030_010955/
│   └── ...
└── session_20251030_011235/
    └── ...
```

## 模型架构

使用 **ResNet18 + LSTM** 时序模型：

- **特征提取**: ResNet18 (ImageNet预训练)
- **时序建模**: 2层LSTM，处理连续帧序列（默认4帧）
- **输入格式**: `[B, T, 3, H, W]` - 批次大小B，序列长度T，3通道RGB，高H宽W
- **输出**: 11维动作向量

## 模型输出

模型输出11维动作向量：
- `[0:4]` - W, A, S, D 移动按键状态（二进制）
- `[4:6]` - mouse_left, mouse_right 鼠标按键状态（二进制）
- `[6:8]` - dx_norm, dy_norm 归一化的鼠标相对位移（连续值，范围[-1,1]）
- `[8:11]` - space, shift, t 特殊按键状态（二进制：刹车、瞄准镜、交互）

## 训练参数

可在 `train_imitation.py` 中调整：

- `BATCH_SIZE = 32` - 批次大小（序列输入建议较小）
- `LEARNING_RATE = 1e-4` - 学习率
- `EPOCHS = 20` - 训练轮数
- `SEQUENCE_LENGTH = 4` - 连续帧序列长度
- `TARGET_W, TARGET_H = 256, 256` - 输入图像尺寸
- `USE_PRETRAINED = True` - 使用ImageNet预训练ResNet18
- `FREEZE_BACKBONE = False` - 是否冻结ResNet18 backbone（可先冻结训练再微调）

## 输出文件

- `tank_imitation_model.pth` - 训练好的模型
- `best_model_epoch_X.pth` - 最佳验证模型
- `training.log` - 训练日志
