@echo off
REM 使用 xmake 构建 World of Tanks AI C++ 模块

echo ====================================
echo Building with xmake
echo ====================================

REM 检查 xmake 是否安装
where xmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: xmake not found!
    echo.
    echo Please install xmake:
    echo.
    echo   Method 1: PowerShell ^(Run as Administrator^)
    echo     powershell -Command "Invoke-Expression (Invoke-Webrequest 'https://xmake.io/psget.txt' -UseBasicParsing).Content"
    echo.
    echo   Method 2: Scoop
    echo     scoop install xmake
    echo.
    echo   Method 3: Manual Download
    echo     https://github.com/xmake-io/xmake/releases
    echo.
    echo After installation, restart this terminal and run this script again.
    echo.
    pause
    exit /b 1
)

echo xmake version:
xmake --version
echo.

REM 检查虚拟环境
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 安装 Python 依赖
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM 配置 xmake
echo Configuring project...
xmake f -c -m release -p windows -a x64

REM 构建
echo Building...
xmake -j %NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ====================================
    echo Build failed!
    echo ====================================
    pause
    exit /b 1
)

REM 安装（复制到 python 目录）
echo Installing...
xmake install -o python

echo.
echo ====================================
echo Build completed successfully!
echo ====================================
echo.
echo C++ bindings have been installed to python/
echo.
echo Next steps:
echo   1. Test installation: python python/tests/test_installation.py
echo   2. Run client: start_windows_client.bat
echo.
pause

