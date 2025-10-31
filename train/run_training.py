#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练启动脚本
简化版训练入口
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train.train_imitation import main as train_main
from train.train_imitation import DATA_ROOT as DEFAULT_DATA_ROOT

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

def resolve_data_dir() -> Path:
    """Resolve data directory with precedence: ENV -> train.DATA_ROOT -> default layout."""
    # 1) ENV override
    env_path = os.environ.get("TRAIN_DATA_ROOT")
    if env_path:
        return Path(env_path)
    # 2) Use DATA_ROOT from train.train_imitation
    if DEFAULT_DATA_ROOT:
        return Path(DEFAULT_DATA_ROOT)
    # 3) Fallback to legacy project layout
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return project_root / "data_collection" / "data" / "recordings"


def check_environment():
    """检查训练环境"""
    logger = logging.getLogger(__name__)
    
    # 检查CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        logger.info(f"当前设备: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA不可用，将使用CPU训练（速度较慢）")
    
    # 检查数据目录（支持 ENV/配置）
    script_dir = Path(__file__).resolve().parent
    data_dir = resolve_data_dir()
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"脚本目录: {script_dir}")
        logger.info("可通过环境变量 TRAIN_DATA_ROOT 覆盖数据目录，例如：")
        logger.info("  set TRAIN_DATA_ROOT=C:/data/recordings  (Windows)")
        return False
    
    # 统计数据
    sessions = list(data_dir.glob("session_*"))
    logger.info(f"找到 {len(sessions)} 个数据session")
    
    for session in sessions:
        actions_file = session / "actions.json"
        frames_dir = session / "frames"
        if actions_file.exists() and frames_dir.exists():
            frame_count = len(list(frames_dir.glob("*.jpg")))
            logger.info(f"  {session.name}: {frame_count} 帧")
    
    return True

def main():
    """主函数"""
    logger = logging.getLogger(__name__)
    logger.info("=== 坦克世界模仿学习训练 ===")
    
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败，退出训练")
        return
    
    # 开始训练
    try:
        train_main()
        logger.info("训练完成！")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
