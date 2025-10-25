# WoT Game Client

游戏客户端，运行在 Windows 环境。

## 功能

- 游戏画面实时捕获
- 键盘鼠标操作控制
- 与训练服务器通信
- 接收并执行 AI 操作指令

## 安装

```bash
cd wot_client
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 架构

- 捕获游戏画面并发送到服务器
- 接收服务器返回的操作指令
- 控制键盘鼠标执行操作

