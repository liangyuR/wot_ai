@echo off
chcp 65001 > nul
title 打包成独立 EXE

echo.
echo ═══════════════════════════════════════════════
echo    坦克世界 AI - 打包成单个 EXE
echo ═══════════════════════════════════════════════
echo.
echo 本脚本将程序打包成一个 exe 文件
echo 打包后无需 Python 环境，双击即可运行
echo.
echo 打包时间约 3-5 分钟，请耐心等待...
echo.
pause

:: 运行简化打包脚本
python build_single_exe.py

if %errorlevel% equ 0 (
    echo.
    echo ═══════════════════════════════════════════════
    echo ✅ 打包成功！
    echo ═══════════════════════════════════════════════
    echo.
    echo 打包文件位于: dist\WoT_DataCollector_Portable.zip
    echo.
    echo 将这个 ZIP 文件发送给你的朋友即可！
    echo.
) else (
    echo.
    echo ═══════════════════════════════════════════════
    echo ❌ 打包失败
    echo ═══════════════════════════════════════════════
    echo.
    echo 请检查错误信息
    echo.
)

pause

