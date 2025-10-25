#!/bin/bash

# WoT AI Training Server 启动脚本 (Linux)

echo "========================================"
echo "WoT AI Training Server (Linux)"
echo "========================================"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "虚拟环境不存在，正在创建..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 启动服务器
echo "启动训练服务器..."
python server_main.py "$@"

