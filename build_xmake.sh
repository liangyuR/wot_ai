#!/bin/bash
# 使用 xmake 构建 World of Tanks AI C++ 模块 (Linux)

echo "======================================"
echo "Building with xmake (Linux)"
echo "======================================"

# 检查 xmake 是否安装
if ! command -v xmake &> /dev/null; then
    echo "ERROR: xmake not found!"
    echo ""
    echo "Please install xmake:"
    echo "  bash <(curl -fsSL https://xmake.io/shget.text)"
    echo ""
    echo "Or download from: https://github.com/xmake-io/xmake/releases"
    exit 1
fi

echo "xmake version:"
xmake --version
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装 Python 依赖
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 配置 xmake
echo "Configuring project..."
xmake f -c -m release -p linux

# 构建
echo "Building..."
xmake -j $(nproc)

if [ $? -ne 0 ]; then
    echo ""
    echo "======================================"
    echo "Build failed!"
    echo "======================================"
    exit 1
fi

# 安装
echo "Installing..."
xmake install -o python

echo ""
echo "======================================"
echo "Build completed successfully!"
echo "======================================"
echo ""
echo "C++ bindings have been installed to python/"
echo ""
echo "Next steps:"
echo "  1. Test installation: python python/tests/test_installation.py"
echo "  2. Run server: ./start_linux_server.sh"
echo ""

