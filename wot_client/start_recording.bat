@echo off
REM WoT Data Collection 启动脚本 (Windows)

echo ========================================
echo WoT Data Collection (Windows)
echo ========================================
echo.

REM 检查虚拟环境
if not exist "venv\" (
    echo [!] 虚拟环境不存在，正在创建...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 无法创建虚拟环境
        echo 请确保已安装 Python 3.7+
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo.
    echo [+] 虚拟环境创建完成
) else (
    call venv\Scripts\activate.bat
)

echo.
echo 可用选项：
echo   1. 开始录制游戏
echo   2. 测试屏幕捕获
echo   3. 运行系统诊断
echo   4. 退出
echo.

set /p choice="请选择 (1-4): "

if "%choice%"=="1" (
    echo.
    echo [+] 启动数据采集...
    python record_main.py %*
) else if "%choice%"=="2" (
    echo.
    echo [+] 测试屏幕捕获...
    python record_main.py --test
) else if "%choice%"=="3" (
    echo.
    echo [+] 运行系统诊断...
    python test_capture.py
) else if "%choice%"=="4" (
    echo 退出...
    exit /b 0
) else (
    echo [!] 无效选择，默认启动录制...
    python record_main.py %*
)

echo.
pause

