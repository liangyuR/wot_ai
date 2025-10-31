@echo off
echo 启动坦克世界模仿学习训练...
echo.

REM 检查Python环境
python --version
if %errorlevel% neq 0 (
    echo 错误: Python未安装或未添加到PATH
    pause
    exit /b 1
)

REM 安装依赖
echo 安装依赖包...
pip install -r requirements.txt

REM 启动训练
echo 开始训练...
python run_training.py

echo 训练完成！
pause
