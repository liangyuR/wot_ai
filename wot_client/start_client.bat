@echo off
REM WoT Game Client 启动脚本 (Windows)

echo ========================================
echo WoT Game Client (Windows)
echo ========================================

REM 检查虚拟环境
if not exist "venv\" (
    echo 虚拟环境不存在，正在创建...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM 启动客户端
echo 启动游戏客户端...
python client_main.py %*

pause

