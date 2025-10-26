#!/bin/bash

# Train WoT AI using Imitation Learning (Behavioral Cloning)
# 使用行为克隆训练坦克世界AI

echo "========================================"
echo "WoT AI - Imitation Learning Training"
echo "========================================"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Default parameters
CONFIG="configs/ppo_config.yaml"
DATA_DIR="../wot_client/data/recordings"
EPOCHS=50
BATCH_SIZE=32
VAL_SPLIT=0.2

# Run training
python train/train_imitation.py \
    --config $CONFIG \
    --data-dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --val-split $VAL_SPLIT \
    "$@"

echo ""
echo "Training completed!"
echo "Check logs in: ./logs/imitation"
echo "Models saved in: ./models/checkpoints"

