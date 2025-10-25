@echo off
REM Windows 版本的同步脚本（使用 WinSCP 或 pscp）

REM 配置
set LINUX_SERVER=user@192.168.1.100
set LINUX_PATH=/home/user/wot_ai

echo ======================================
echo Syncing to Linux Server
echo Server: %LINUX_SERVER%
echo Path: %LINUX_PATH%
echo ======================================

REM 检查 pscp 是否安装
where pscp >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pscp not found!
    echo Please install PuTTY or use WinSCP manually.
    echo Download: https://www.chiark.greenend.org.uk/~sgtatham/putty/
    pause
    exit /b 1
)

REM 同步 Python 代码
echo Syncing Python code...
pscp -r -batch python %LINUX_SERVER%:%LINUX_PATH%/

REM 同步配置
echo Syncing configs...
pscp -r -batch configs %LINUX_SERVER%:%LINUX_PATH%/

REM 同步依赖
echo Syncing requirements...
pscp -batch requirements.txt %LINUX_SERVER%:%LINUX_PATH%/

echo ======================================
echo Sync completed!
echo ======================================
echo.
echo Alternative: Use WinSCP GUI
echo 1. Download WinSCP: https://winscp.net/
echo 2. Connect to %LINUX_SERVER%
echo 3. Upload python/, configs/, requirements.txt
echo.
pause

