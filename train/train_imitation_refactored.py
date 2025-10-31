#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克世界行为克隆训练脚本（重构版）
使用模块化架构和性能优化
"""

import torch
import os
import logging
from pathlib import Path

from core.constants import (
    DATA_ROOT, MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, VAL_SPLIT,
    NUM_WORKERS, PREFETCH_FACTOR, SEQUENCE_LENGTH
)
from data import TankImitationDataset
from models import TankImitationNet
from training import ImitationTrainer, ModelCheckpoint, LoggingCallback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_data_loaders(data_root: str = DATA_ROOT, batch_size: int = BATCH_SIZE,
                       val_split: float = VAL_SPLIT):
    """创建优化的数据加载器"""
    from torch.utils.data import DataLoader
    
    # 创建数据集（分辨率将从meta.json自动读取）
    dataset = TankImitationDataset(data_root, sequence_length=SEQUENCE_LENGTH)
    
    # 划分训练集和验证集
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoader性能选项
    use_cuda = torch.cuda.is_available()
    num_workers = NUM_WORKERS
    pin_memory = use_cuda
    persistent_workers = use_cuda and num_workers > 0
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=PREFETCH_FACTOR if persistent_workers else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=PREFETCH_FACTOR if persistent_workers else None,
    ) if val_size > 0 else None
    
    logger.info(f"训练集: {train_size} 样本, 验证集: {val_size} 样本")
    
    return train_loader, val_loader


def main():
    """主函数"""
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 检查数据目录
    if not os.path.exists(DATA_ROOT):
        logger.error(f"数据目录不存在: {DATA_ROOT}")
        return
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(DATA_ROOT)
    
    # 创建模型
    model = TankImitationNet()
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建回调
    callbacks = [
        LoggingCallback(log_interval=10),
        ModelCheckpoint(
            save_path=MODEL_SAVE_PATH.replace('.pth', ''),
            monitor='val_loss',
            mode='min',
            save_best=True,
            save_last=True
        )
    ]
    
    # 创建训练器
    trainer = ImitationTrainer(
        model, train_loader, val_loader, device,
        callbacks=callbacks
    )
    
    # 开始训练
    trainer.train(EPOCHS)
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()

