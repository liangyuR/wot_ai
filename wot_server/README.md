# WoT AI Training Server

深度学习训练服务器，运行在 Linux 环境。

## 功能

- 强化学习模型训练
- 目标检测模型训练 (YOLOv8)
- 提供 REST API 和 WebSocket 接口供客户端调用
- 模型推理服务

## 安装

```bash
cd wot_server
python -m venv venv
source venv/bin/activate  # Linux
pip install -r requirements.txt
```

## 架构

- 接收客户端发送的游戏画面
- 使用训练好的模型进行推理
- 返回操作指令给客户端

