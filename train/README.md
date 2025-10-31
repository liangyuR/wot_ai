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

## 模型输出

模型输出8维动作向量：
- `[0:4]` - W, A, S, D 按键状态
- `[4:6]` - 鼠标左键、右键状态  
- `[6:8]` - 归一化的鼠标X、Y坐标

## 训练参数

可在 `train_imitation.py` 中调整：

- `BATCH_SIZE = 32` - 批次大小
- `LEARNING_RATE = 1e-4` - 学习率
- `EPOCHS = 20` - 训练轮数
- `TARGET_W, TARGET_H = 84, 84` - 输入图像尺寸

## 输出文件

- `tank_imitation_model.pth` - 训练好的模型
- `best_model_epoch_X.pth` - 最佳验证模型
- `training.log` - 训练日志
