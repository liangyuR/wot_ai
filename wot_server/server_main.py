"""
WoT AI Training Server - Main Entry Point
Linux 端训练服务器主入口
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from network.training_server import main

if __name__ == "__main__":
    main()

