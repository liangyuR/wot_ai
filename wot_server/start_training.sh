#!/bin/bash

# WoT AI 模型训练脚本 (Linux)

echo "========================================"
echo "WoT AI Model Training (Linux)"
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

# 开始训练
echo "开始模型训练..."
python train_main.py "$@"

