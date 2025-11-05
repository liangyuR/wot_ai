#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备工具：准备小地图检测训练数据
"""

from pathlib import Path
import shutil
import yaml
try:
    from path_planning.utils.logger import SetupLogger
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.logger import SetupLogger

logger = SetupLogger(__name__)


def PrepareDataset(image_dir: str, output_dir: str, classes: list = None) -> bool:
    """
    准备数据集目录结构
    
    Args:
        image_dir: 图像目录
        output_dir: 输出目录
        classes: 类别列表，如果为None则使用默认类别
    
    Returns:
        是否成功
    """
    if classes is None:
        classes = [
            'minimap_self',
            'minimap_ally',
            'minimap_enemy',
            'flag_base',
            'minimap_obstacle',
            'minimap_road'
        ]
    
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    
    if not image_path.exists():
        logger.error(f"图像目录不存在: {image_dir}")
        return False
    
    # 创建输出目录结构
    output_path.mkdir(parents=True, exist_ok=True)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # 创建data.yaml配置文件
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    with open(output_path / "data.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    
    logger.info(f"数据集目录结构已创建: {output_path}")
    logger.info("请将图像和标注文件复制到 train/images 和 val/images 目录")
    logger.info("标注文件格式：YOLO格式（每张图像对应一个.txt文件）")
    
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备小地图检测数据集")
    parser.add_argument("--image_dir", type=str, required=True, help="图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    args = parser.parse_args()
    
    PrepareDataset(args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()

