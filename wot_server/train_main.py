"""
WoT AI Training Script
模型训练主脚本
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train.train_ppo import main

if __name__ == "__main__":
    main()

