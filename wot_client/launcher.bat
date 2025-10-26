@echo off
chcp 65001 > nul
title 坦克世界 AI - 数据采集工具

echo.
echo ========================================
echo   坦克世界 AI 数据采集工具
echo ========================================
echo.

:: 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 需要管理员权限！
    echo.
    echo 请右键点击此文件，选择"以管理员身份运行"
    echo.
    echo 没有管理员权限将无法正常录制键盘输入！
    echo.
    pause
    exit /b 1
)

echo [信息] 管理员权限检查通过
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo [错误] 找不到虚拟环境！
    echo.
    echo 请确保解压完整，或重新下载安装包。
    echo.
    pause
    exit /b 1
)

echo [信息] 激活 Python 虚拟环境...
call venv\Scripts\activate.bat

echo [信息] 启动配置界面...
echo.

:: 启动配置 GUI
python config_gui.py

:: 如果 GUI 异常退出
if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序异常退出
    echo.
    pause
)

