@echo off
REM Train WoT AI using Imitation Learning (Behavioral Cloning)
REM 使用行为克隆训练坦克世界AI

echo ========================================
echo WoT AI - Imitation Learning Training
echo ========================================

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Default parameters
set CONFIG=configs/ppo_config.yaml
set DATA_DIR=..\wot_client\data\recordings
set EPOCHS=50
set BATCH_SIZE=32
set VAL_SPLIT=0.2

REM Run training
python train\train_imitation.py ^
    --config %CONFIG% ^
    --data-dir %DATA_DIR% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --val-split %VAL_SPLIT% ^
    %*

echo.
echo Training completed!
echo Check logs in: .\logs\imitation
echo Models saved in: .\models\checkpoints
pause

