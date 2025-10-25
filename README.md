# World of Tanks AI Project

基于深度学习和强化学习的坦克世界 AI 项目，采用客户端-服务器架构。

## 项目结构

```
wot_ai/
├── wot_client/          # Windows 游戏客户端（完整独立）
│   ├── cpp/             # C++ 绑定（屏幕捕获、输入控制）
│   ├── cpp_bindings/    # 编译后的 Python 绑定模块
│   ├── configs/         # 客户端配置文件
│   ├── network/         # 网络通信模块
│   ├── data_collection/ # 数据采集工具
│   ├── utils/           # 工具函数
│   ├── requirements.txt # 客户端依赖
│   ├── start_client.bat # 启动脚本
│   └── README.md        # 客户端文档
│
├── wot_server/          # Linux 训练服务端（完整独立）
│   ├── agents/          # RL 智能体实现
│   ├── env/             # Gymnasium 环境
│   ├── train/           # 训练脚本
│   ├── vision/          # 目标检测模块
│   ├── network/         # 服务端网络通信
│   ├── configs/         # 服务端配置文件
│   ├── utils/           # 工具函数
│   ├── requirements.txt # 服务端依赖
│   ├── start_server.sh  # 启动脚本
│   └── README.md        # 服务端文档
│
├── docs/                # 项目文档
├── scripts/             # 辅助脚本（同步等）
├── xmake.lua            # C++ 构建配置（xmake）
├── build_xmake.bat      # Windows 构建脚本
├── build_xmake.sh       # Linux 构建脚本
├── PROJECT_OVERVIEW.md  # 项目概览
├── GETTING_STARTED.md   # 快速入门指南
└── README.md            # 项目主文档（本文件）
```

## 架构设计

### wot_server (Linux)
- **作用**: 模型训练和推理服务
- **环境**: Linux (用于 GPU 训练)
- **技术栈**: PyTorch, Stable-Baselines3, YOLOv8, FastAPI
- **功能**:
  - 强化学习模型训练
  - 目标检测模型训练
  - 提供 REST API 和 WebSocket 服务
  - 实时推理返回操作指令

### wot_client (Windows)
- **作用**: 游戏自动化客户端
- **环境**: Windows (游戏运行平台)
- **技术栈**: pywin32, mss, pynput, WebSocket
- **功能**:
  - 实时捕获游戏画面
  - 控制键盘鼠标操作
  - 与服务器通信传输数据
  - 执行 AI 指令

## 通信协议

客户端和服务器之间通过 WebSocket 进行实时通信：
1. 客户端捕获游戏画面 → 发送到服务器
2. 服务器进行模型推理 → 返回操作指令
3. 客户端执行操作 → 继续循环

## 快速开始

### 1. Windows 客户端设置

#### 1.1 编译 C++ 绑定模块
```cmd
REM 安装 xmake（如果未安装）
install_xmake.bat

REM 编译 C++ 绑定
build_xmake.bat
```

编译后的 `cpp_bindings.pyd` 将输出到 `wot_client/cpp_bindings/` 目录。

#### 1.2 安装 Python 依赖
```cmd
cd wot_client
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### 1.3 启动客户端
```cmd
REM 使用启动脚本（会自动配置）
start_client.bat

REM 或直接运行
python client_main.py
```

详细说明请参考：`wot_client/README.md`

### 2. Linux 服务端设置

#### 2.1 安装 Python 依赖
```bash
cd wot_server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2.2 启动训练服务器
```bash
# 使用启动脚本
./start_server.sh

# 或直接运行
python server_main.py
```

详细说明请参考：`wot_server/README.md`

### 3. 开发者工具

- **同步脚本**: `scripts/sync_to_linux.*` - 将代码同步到 Linux 服务器
- **项目文档**: `docs/` 目录包含详细的架构和部署文档
