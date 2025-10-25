@echo off
REM xmake 一键安装脚本（Windows）

echo ====================================
echo Installing xmake
echo ====================================
echo.

REM 检查管理员权限
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: This script requires administrator privileges.
    echo Please run as Administrator.
    echo.
    pause
    exit /b 1
)

echo Installing xmake via PowerShell...
echo This may take a few minutes...
echo.

powershell -Command "& {Invoke-Expression (Invoke-Webrequest 'https://xmake.io/psget.txt' -UseBasicParsing).Content}"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================
    echo xmake installed successfully!
    echo ====================================
    echo.
    echo Please restart your terminal and run:
    echo   build_xmake.bat
    echo.
) else (
    echo.
    echo ====================================
    echo Installation failed!
    echo ====================================
    echo.
    echo Alternative installation methods:
    echo.
    echo 1. Scoop:
    echo    scoop install xmake
    echo.
    echo 2. Manual download:
    echo    https://github.com/xmake-io/xmake/releases
    echo.
)

pause

