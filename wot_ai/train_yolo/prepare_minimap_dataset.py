#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测数据集准备脚本
从 label_date 目录自动准备 YOLO 训练数据集
"""

import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional
import yaml

# 统一导入机制
from loguru import logger
from wot_ai.utils.paths import setup_python_path, resolve_path, ensure_dir
from wot_ai.train_yolo.config_loader import LoadClassesFromConfig
setup_python_path()

def MatchImageLabelPairs(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    """
    匹配图像和标签文件对（文件名完全相同，仅扩展名不同）
    
    Args:
        image_dir: 图像目录
        label_dir: 标签目录
    
    Returns:
        (图像路径, 标签路径) 的列表
    """
    pairs = []
    
    # 遍历所有图像文件，查找对应的标签文件
    for img_file in image_dir.glob("*.png"):
        # 获取文件名（不含扩展名）
        img_stem = img_file.stem
        # 查找对应的标签文件
        label_file = label_dir / f"{img_stem}.txt"
        
        if label_file.exists():
            pairs.append((img_file, label_file))
        else:
            logger.warning(f"图像 {img_file.name} 没有对应的标签文件: {label_file.name}")
    
    # 检查是否有标签文件没有对应的图像
    for label_file in label_dir.glob("*.txt"):
        label_stem = label_file.stem
        img_file = image_dir / f"{label_stem}.png"
        if not img_file.exists():
            logger.warning(f"标签 {label_file.name} 没有对应的图像文件: {img_file.name}")
    
    return pairs


def PrepareMinimapDataset(
    source_dir: Optional[Path] = None,
    output_dir: Path = None,
    source_images: Optional[Path] = None,
    source_labels: Optional[Path] = None,
    train_ratio: float = 0.8,
    random_seed: Optional[int] = None,
    config_path: Optional[Path] = None
) -> bool:
    """
    准备小地图检测数据集
    
    Args:
        source_dir: 源数据目录（包含 images 和 labels 子目录，当 source_images 和 source_labels 未指定时使用）
        output_dir: 输出数据集目录
        source_images: 源图像目录（如果指定则优先使用）
        source_labels: 源标签目录（如果指定则优先使用）
        train_ratio: 训练集比例（默认 0.8）
        random_seed: 随机种子（用于可重复分割）
        config_path: 配置文件路径
    
    Returns:
        是否成功
    """
    # 优先使用直接指定的路径
    if source_images is not None and source_labels is not None:
        image_dir = Path(source_images)
        label_dir = Path(source_labels)
    elif source_dir is not None:
        image_dir = source_dir / "images"
        label_dir = source_dir / "labels"
    else:
        logger.error("必须指定 source_dir 或同时指定 source_images 和 source_labels")
        return False
    
    if not image_dir.exists():
        logger.error(f"图像目录不存在: {image_dir}")
        return False
    
    if not label_dir.exists():
        logger.error(f"标签目录不存在: {label_dir}")
        return False
    
    # 匹配图像和标签对
    logger.info("正在匹配图像和标签文件...")
    pairs = MatchImageLabelPairs(image_dir, label_dir)
    
    if len(pairs) == 0:
        logger.error("未找到匹配的图像-标签对")
        return False
    
    logger.info(f"找到 {len(pairs)} 对匹配的图像-标签文件")
    
    # 随机分割数据集
    if random_seed is not None:
        random.seed(random_seed)
    
    random.shuffle(pairs)
    train_size = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]
    
    logger.info(f"数据集分割: 训练集 {len(train_pairs)} 对, 验证集 {len(val_pairs)} 对")
    
    # 创建输出目录结构（使用统一路径管理）
    ensure_dir(output_dir)
    train_img_dir = ensure_dir(output_dir / "images" / "train")
    train_label_dir = ensure_dir(output_dir / "labels" / "train")
    val_img_dir = ensure_dir(output_dir / "images" / "val")
    val_label_dir = ensure_dir(output_dir / "labels" / "val")
    
    # 复制文件
    logger.info("正在复制训练集文件...")
    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        shutil.copy2(label_path, train_label_dir / label_path.name)
    
    logger.info("正在复制验证集文件...")
    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        shutil.copy2(label_path, val_label_dir / label_path.name)
    
    # 生成 dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    class_names = LoadClassesFromConfig(config_path)
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)
    
    logger.info(f"数据集准备完成: {output_dir}")
    logger.info(f"数据集配置文件: {yaml_path}")
    logger.info(f"训练集: {len(train_pairs)} 对")
    logger.info(f"验证集: {len(val_pairs)} 对")
    
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备小地图检测数据集")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="源数据目录（包含 images 和 labels 子目录）"
    )
    parser.add_argument(
        "--source_images",
        type=str,
        default=None,
        help="源图像目录（如果指定则优先使用）"
    )
    parser.add_argument(
        "--source_labels",
        type=str,
        default=None,
        help="源标签目录（如果指定则优先使用）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/processed/minimap",
        help="输出数据集目录"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认 0.8）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可重复分割）"
    )
    
    args = parser.parse_args()
    
    # 解析路径（使用统一路径管理）
    source_dir = None
    source_images = None
    source_labels = None
    
    if args.source_images and args.source_labels:
        source_images = resolve_path(args.source_images)
        source_labels = resolve_path(args.source_labels)
    elif args.source:
        source_dir = resolve_path(args.source)
    else:
        parser.error("必须指定 --source 或同时指定 --source_images 和 --source_labels")
    
    output_dir = resolve_path(args.output)
    
    success = PrepareMinimapDataset(
        source_dir=source_dir,
        source_images=source_images,
        source_labels=source_labels,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

