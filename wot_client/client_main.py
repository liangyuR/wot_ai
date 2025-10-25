"""
WoT Game Client - Main Entry Point
Windows 端游戏客户端主入口
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from network.game_client import main

if __name__ == "__main__":
    main()

