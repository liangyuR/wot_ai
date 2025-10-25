# 系统架构详解

## 整体架构

### 方案 A：分布式架构（推荐）

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          整体系统架构                                      │
└───────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐         ┌──────────────────────────────────┐
│     Windows PC (游戏端)          │  TCP    │   Linux Server (训练端)          │
│                                  │  9999   │                                  │
│  ┌────────────────────────────┐ │  ◄───►  │  ┌────────────────────────────┐  │
│  │  World of Tanks Client     │ │         │  │  Training Server           │  │
│  │  - 1920x1080 @ 60fps       │ │         │  │  - Model Inference         │  │
│  └──────────┬─────────────────┘ │         │  │  - Data Collection         │  │
│             │                    │         │  │  - Online Learning         │  │
│  ┌──────────▼─────────────────┐ │         │  └──────────┬─────────────────┘  │
│  │  Screen Capture (C++)      │ │         │             │                    │
│  │  - DirectX/GDI             │ │         │  ┌──────────▼─────────────────┐  │
│  │  - 60+ FPS                 │ │         │  │  PPO Agent                 │  │
│  └──────────┬─────────────────┘ │         │  │  - CNN Feature Extractor   │  │
│             │                    │         │  │  - Policy Network          │  │
│  ┌──────────▼─────────────────┐ │         │  │  - Value Network           │  │
│  │  Game Client (Python)      │ │         │  └──────────┬─────────────────┘  │
│  │  - Frame Encoding (JPEG)   │ │────────►│             │                    │
│  │  - Network Send            │ │ Frames  │  ┌──────────▼─────────────────┐  │
│  │  - Action Receive          │ │◄────────│  │  Observation Processing    │  │
│  └──────────┬─────────────────┘ │ Actions │  │  - Frame Preprocessing     │  │
│             │                    │         │  │  - State Extraction        │  │
│  ┌──────────▼─────────────────┐ │         │  └──────────┬─────────────────┘  │
│  │  Input Control (C++)       │ │         │             │                    │
│  │  - Keyboard Simulation     │ │         │  ┌──────────▼─────────────────┐  │
│  │  - Mouse Simulation        │ │         │  │  GPU Training (RTX 5090)   │  │
│  │  - SendInput API           │ │         │  │  - PyTorch + CUDA          │  │
│  └────────────────────────────┘ │         │  │  - TensorBoard Logging     │  │
│                                  │         │  └────────────────────────────┘  │
└─────────────────────────────────┘         └──────────────────────────────────┘
```

## 模块详解

### 1. 屏幕捕获模块（C++）

**文件**: `cpp/screen_capture/`

**功能**：
- 高性能屏幕捕获（Windows GDI/DirectX）
- 目标：60+ FPS
- 输出：RGB 图像缓冲区

**技术细节**：
```cpp
class ScreenCapture {
 public:
  ScreenCapture(int width, int height);
  std::vector<uint8_t> Capture();  // 返回 width*height*3 字节
  double GetFps() const;
};
```

**性能优化**：
- DirectX Desktop Duplication API（推荐）
- GDI BitBlt（兼容方案）
- 零拷贝内存映射

### 2. 输入控制模块（C++）

**文件**: `cpp/input_control/`

**功能**：
- 模拟键盘输入（WASD 移动）
- 模拟鼠标操作（瞄准、射击）
- Windows SendInput API

**技术细节**：
```cpp
class InputControl {
 public:
  void PressKey(char key);
  void ReleaseKey(char key);
  void MoveMouse(int dx, int dy);
  void MouseClick(MouseButton button);
};
```

**安全措施**：
- 输入频率限制
- 防止误操作的安全模式
- 紧急停止机制

### 3. 游戏客户端（Python）

**文件**: `python/network/game_client.py`

**功能**：
- 调用 C++ 模块捕获屏幕
- JPEG 压缩后通过网络发送
- 接收动作指令并执行

**数据流**：
```python
frame (1920x1080x3) 
  → JPEG compress (80% quality)
  → pickle serialize
  → TCP socket send (8-15 Mbps)
```

**关键参数**：
- FPS: 30（可调节 10-60）
- 压缩质量: 80%
- 延迟: <50ms（局域网）

### 4. 训练服务器（Python）

**文件**: `python/network/training_server.py`

**功能**：
- 接收游戏帧
- AI 模型推理
- 发送动作指令
- 在线数据收集

**推理流程**：
```python
frame → preprocess → CNN → policy network → action
```

**性能**：
- 推理延迟: <10ms @RTX 5090
- 吞吐量: 100+ FPS

### 5. PPO 智能体（Python）

**文件**: `python/agents/ppo_agent.py`

**架构**：

```
Observation:
  - Screen: (84, 84, 4) stacked frames
  - Health: (1,) float
  - Ammo: (1,) float
  - Minimap: (64, 64, 3)

         ↓
         
CNN Feature Extractor:
  Conv2d(4, 32, 8, 4) → ReLU
  Conv2d(32, 64, 4, 2) → ReLU
  Conv2d(64, 64, 3, 1) → ReLU
  Flatten → (3136,)
  
         ↓
         
Fully Connected:
  Linear(3136 + minimap_features + 2, 512) → ReLU
  
         ↓           ↓
         
  Policy Head    Value Head
  Linear(512, 256)  Linear(512, 256)
  ReLU              ReLU
  Linear(256, 8)    Linear(256, 1)
  Softmax           -
  
         ↓           ↓
         
    Action      State Value
```

**动作空间**：
- 0: 前进
- 1: 后退
- 2: 左转
- 3: 右转
- 4: 炮塔左转
- 5: 炮塔右转
- 6: 射击
- 7: 无操作

### 6. 计算机视觉模块（Python）

**文件**: `python/vision/detector.py`

**功能**：
- YOLOv8 目标检测
- 识别敌方坦克
- 自动瞄准辅助

**检测流程**：
```python
frame → YOLOv8 → detections → closest_target → aim_offset
```

**性能**：
- YOLOv8n: ~5ms @RTX 5090
- YOLOv8x: ~15ms @RTX 5090
- mAP@0.5: 0.85+（训练后）

## 训练流程

### 在线训练（推荐）

```
1. Windows Client 采集数据
   ↓
2. 发送到 Linux Server
   ↓
3. Server 缓存数据
   ↓
4. 达到 batch_size 后训练
   ↓
5. 更新模型权重
   ↓
6. 使用新模型推理
   ↓
7. 返回步骤 1
```

### 离线训练

```
1. 收集数据 (record_gameplay.py)
   ↓
2. 保存到磁盘 (data/recordings/)
   ↓
3. 预处理数据
   ↓
4. 训练模型 (train_ppo.py)
   ↓
5. 评估模型
   ↓
6. 部署模型
```

## 网络协议

### 数据包格式

**Frame 数据包（Client → Server）**:
```python
{
  'frame': bytes,           # JPEG 压缩的图像
  'shape': tuple,           # (H, W, C)
  'game_state': {
    'health': float,        # 0.0-1.0
    'ammo': int,            # 弹药数量
    'position': (x, y),     # 坐标（可选）
  },
  'timestamp': float        # Unix 时间戳
}
```

**Action 数据包（Server → Client）**:
```python
{
  'move': str,              # 'none', 'forward', 'backward'
  'turn': str,              # 'none', 'left', 'right'
  'turret': str,            # 'none', 'left', 'right'
  'shoot': bool,            # True/False
  'timestamp': float
}
```

### 传输协议

- **协议**: TCP
- **端口**: 9999（可配置）
- **序列化**: pickle
- **压缩**: JPEG（图像）

### 错误处理

- 连接断开：自动重连
- 超时：5 秒超时
- 数据损坏：跳过并记录

## 性能基准

### Windows Client

| 组件 | 性能 |
|------|------|
| 屏幕捕获 | 60+ FPS |
| JPEG 压缩 | ~5ms |
| 网络发送 | ~2ms (LAN) |
| **总延迟** | **<20ms** |

### Linux Server

| 组件 | 性能 |
|------|------|
| 网络接收 | ~2ms |
| 图像解码 | ~3ms |
| CNN 推理 | ~8ms @5090 |
| 动作生成 | ~1ms |
| **总延迟** | **<15ms** |

### 端到端延迟

- **局域网**: 30-50ms
- **互联网**: 100-200ms（取决于带宽）

### 训练性能（RTX 5090）

| 配置 | 吞吐量 | 时间（1000万步）|
|------|--------|----------------|
| 单客户端 | 5K steps/s | ~30 小时 |
| 4 客户端并行 | 18K steps/s | ~8 小时 |

## 扩展性设计

### 多客户端支持

```python
# Server 端
clients = []
for i in range(4):
    client = accept_connection()
    thread = Thread(target=handle_client, args=(client,))
    thread.start()
```

### 云端部署

支持部署到：
- AWS EC2 (p4d.24xlarge - 8x A100)
- Google Cloud (a2-highgpu-8g)
- Azure NC 系列

### 分布式训练

```yaml
# 使用 Ray 或 PyTorch DDP
training:
  distributed: true
  num_gpus: 8
  backend: "nccl"
```

## 安全考虑

### 反作弊检测

**风险**：
- 内存读取检测
- 输入模式识别
- 异常行为检测

**缓解措施**：
- 模拟人类输入节奏
- 添加随机延迟
- 避免完美操作

### 数据隐私

- 不收集个人信息
- 本地数据处理
- 可选的数据加密

## 故障排查

### 常见问题

**1. 延迟过高**
- 降低分辨率
- 降低 FPS
- 提高 JPEG 压缩率

**2. GPU 未使用**
- 检查 CUDA 安装
- 验证 PyTorch 版本
- 检查 GPU 驱动

**3. 连接失败**
- 检查防火墙
- 验证 IP 地址
- 测试端口连通性

## 总结

这个架构设计的核心理念：

1. **职责分离**：游戏交互和 AI 训练完全解耦
2. **性能优化**：C++ 处理性能关键路径，Python 处理 AI 逻辑
3. **灵活部署**：支持本地和云端，支持单机和分布式
4. **可扩展性**：模块化设计，易于添加新功能
5. **实用性**：兼顾学习价值和实际可用性

