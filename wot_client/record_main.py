"""
WoT Data Collection Script
游戏数据采集脚本
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.record_gameplay import main

if __name__ == "__main__":
    main()

