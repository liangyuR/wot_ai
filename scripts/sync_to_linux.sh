#!/bin/bash
# 从 Windows 同步代码到 Linux 服务器

# 配置
LINUX_SERVER="user@192.168.1.100"
LINUX_PATH="~/wot_ai"

echo "======================================"
echo "Syncing to Linux Server"
echo "Server: $LINUX_SERVER"
echo "Path: $LINUX_PATH"
echo "======================================"

# 同步 Python 代码
echo "Syncing Python code..."
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv' \
    --exclude='.git' \
    python/ $LINUX_SERVER:$LINUX_PATH/python/

# 同步配置文件
echo "Syncing configs..."
rsync -avz --progress configs/ $LINUX_SERVER:$LINUX_PATH/configs/

# 同步依赖文件
echo "Syncing requirements..."
scp requirements.txt $LINUX_SERVER:$LINUX_PATH/

# 同步文档
echo "Syncing docs..."
rsync -avz --progress docs/ $LINUX_SERVER:$LINUX_PATH/docs/

echo "======================================"
echo "Sync completed!"
echo "======================================"
echo ""
echo "Next steps on Linux server:"
echo "  ssh $LINUX_SERVER"
echo "  cd $LINUX_PATH"
echo "  source venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  python python/network/training_server.py"

