"""
Imitation Learning / Behavioral Cloning Training Script
行为克隆训练脚本 - 监督学习方式训练AI模仿人类玩家
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import WotRecordingDataset
from data.action_mapper import ActionMapper


class ImitationCNN(nn.Module):
    """
    行为克隆CNN网络
    
    输入：游戏画面 (256x256x3)
    输出：动作概率分布 [move(3), turn(5), turret(5), shoot(2)]
    """
    
    def __init__(self, action_dims: list):
        """
        初始化网络
        
        Args:
            action_dims: 动作维度列表 [3, 5, 5, 2]
        """
        super().__init__()
        
        self.action_dims_ = action_dims
        
        # CNN特征提取器（类似Nature DQN）
        self.features_ = nn.Sequential(
            # Conv1: 256x256x3 -> 62x62x32
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Conv2: 62x62x32 -> 30x30x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Conv3: 30x30x64 -> 28x28x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Conv4: 28x28x64 -> 14x14x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Flatten()
        )
        
        # 计算展平后的特征维度
        with torch.no_grad():
            sample = torch.zeros(1, 3, 256, 256)
            feature_dim = self.features_(sample).shape[1]
            
        # 全连接层
        self.fc_common_ = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 为每个动作维度创建独立的输出头
        self.move_head_ = nn.Linear(512, action_dims[0])
        self.turn_head_ = nn.Linear(512, action_dims[1])
        self.turret_head_ = nn.Linear(512, action_dims[2])
        self.shoot_head_ = nn.Linear(512, action_dims[3])
        
        logger.info(f"ImitationCNN initialized:")
        logger.info(f"  - Feature dim: {feature_dim}")
        logger.info(f"  - Action dims: {action_dims}")
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            (move_logits, turn_logits, turret_logits, shoot_logits)
        """
        # 特征提取
        features = self.features_(x)
        
        # 公共特征
        common = self.fc_common_(features)
        
        # 各个动作头
        move_logits = self.move_head_(common)
        turn_logits = self.turn_head_(common)
        turret_logits = self.turret_head_(common)
        shoot_logits = self.shoot_head_(common)
        
        return move_logits, turn_logits, turret_logits, shoot_logits
        

class ImitationTrainer:
    """行为克隆训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config_ = config
        
        # 设备
        self.device_ = torch.device(
            config.get('training', {}).get('device', 'cuda') 
            if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device_}")
        
        # 动作映射器
        self.action_mapper_ = ActionMapper(config)
        action_dims = self.action_mapper_.GetActionDims()
        
        # 创建模型
        self.model_ = ImitationCNN(action_dims).to(self.device_)
        
        # 优化器
        lr = config.get('imitation', {}).get('learning_rate', 1e-4)
        weight_decay = config.get('imitation', {}).get('weight_decay', 1e-5)
        self.optimizer_ = optim.Adam(
            self.model_.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler_ = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 损失函数（交叉熵）
        self.criterion_ = nn.CrossEntropyLoss()
        
        # TensorBoard
        log_dir = config.get('training', {}).get('tensorboard_log', './logs/imitation')
        self.writer_ = SummaryWriter(log_dir)
        
        # 训练统计
        self.global_step_ = 0
        self.best_val_loss_ = float('inf')
        
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int
    ):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
        """
        logger.info("=" * 80)
        logger.info("开始行为克隆训练")
        logger.info("=" * 80)
        logger.info(f"训练样本: {len(train_loader.dataset)}")
        logger.info(f"验证样本: {len(val_loader.dataset)}")
        logger.info(f"训练轮数: {epochs}")
        logger.info("=" * 80)
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练阶段
            train_loss, train_acc = self.trainEpoch(train_loader)
            
            # 验证阶段
            val_loss, val_acc = self.validateEpoch(val_loader)
            
            # 学习率调度
            self.scheduler_.step(val_loss)
            
            # 记录到TensorBoard
            self.writer_.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            
            self.writer_.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss_:
                self.best_val_loss_ = val_loss
                self.saveModel('best_imitation_model.pth')
                logger.info(f"✓ 保存最佳模型 (val_loss={val_loss:.4f})")
                
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.saveModel(f'checkpoint_epoch_{epoch+1}.pth')
                
        logger.info("\n训练完成！")
        self.saveModel('final_imitation_model.pth')
        
    def trainEpoch(self, train_loader: DataLoader) -> tuple:
        """
        训练一个epoch
        
        Returns:
            (average_loss, average_accuracy)
        """
        self.model_.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, actions) in enumerate(pbar):
            # 转换数据
            images = self.prepareImages(images)
            move_targets, turn_targets, turret_targets, shoot_targets = self.prepareActions(actions)
            
            # 前向传播
            move_logits, turn_logits, turret_logits, shoot_logits = self.model_(images)
            
            # 计算损失
            loss_move = self.criterion_(move_logits, move_targets)
            loss_turn = self.criterion_(turn_logits, turn_targets)
            loss_turret = self.criterion_(turret_logits, turret_targets)
            loss_shoot = self.criterion_(shoot_logits, shoot_targets)
            
            # 总损失（加权平均）
            loss = (loss_move + loss_turn + loss_turret + loss_shoot) / 4.0
            
            # 反向传播
            self.optimizer_.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
            
            self.optimizer_.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率（所有维度都正确才算正确）
            with torch.no_grad():
                move_pred = move_logits.argmax(dim=1)
                turn_pred = turn_logits.argmax(dim=1)
                turret_pred = turret_logits.argmax(dim=1)
                shoot_pred = shoot_logits.argmax(dim=1)
                
                correct = (
                    (move_pred == move_targets) &
                    (turn_pred == turn_targets) &
                    (turret_pred == turret_targets) &
                    (shoot_pred == shoot_targets)
                ).sum().item()
                
                total_correct += correct
                total_samples += images.size(0)
                
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': total_correct / total_samples
            })
            
            # TensorBoard
            self.writer_.add_scalar('Train/BatchLoss', loss.item(), self.global_step_)
            self.global_step_ += 1
            
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        logger.info(f"  Train - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        return avg_loss, avg_acc
        
    def validateEpoch(self, val_loader: DataLoader) -> tuple:
        """
        验证一个epoch
        
        Returns:
            (average_loss, average_accuracy)
        """
        self.model_.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, actions in tqdm(val_loader, desc="Validation"):
                # 转换数据
                images = self.prepareImages(images)
                move_targets, turn_targets, turret_targets, shoot_targets = self.prepareActions(actions)
                
                # 前向传播
                move_logits, turn_logits, turret_logits, shoot_logits = self.model_(images)
                
                # 计算损失
                loss_move = self.criterion_(move_logits, move_targets)
                loss_turn = self.criterion_(turn_logits, turn_targets)
                loss_turret = self.criterion_(turret_logits, turret_targets)
                loss_shoot = self.criterion_(shoot_logits, shoot_targets)
                
                loss = (loss_move + loss_turn + loss_turret + loss_shoot) / 4.0
                
                total_loss += loss.item()
                
                # 计算准确率
                move_pred = move_logits.argmax(dim=1)
                turn_pred = turn_logits.argmax(dim=1)
                turret_pred = turret_logits.argmax(dim=1)
                shoot_pred = shoot_logits.argmax(dim=1)
                
                correct = (
                    (move_pred == move_targets) &
                    (turn_pred == turn_targets) &
                    (turret_pred == turret_targets) &
                    (shoot_pred == shoot_targets)
                ).sum().item()
                
                total_correct += correct
                total_samples += images.size(0)
                
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_correct / total_samples
        
        logger.info(f"  Val   - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        return avg_loss, avg_acc
        
    def prepareImages(self, images: np.ndarray) -> torch.Tensor:
        """
        准备图像数据
        
        Args:
            images: numpy数组 (B, H, W, 3)
            
        Returns:
            torch张量 (B, 3, H, W)
        """
        # 转换为tensor
        images = torch.from_numpy(images).float()
        
        # (B, H, W, 3) -> (B, 3, H, W)
        images = images.permute(0, 3, 1, 2)
        
        # 归一化到[0, 1]
        images = images / 255.0
        
        return images.to(self.device_)
        
    def prepareActions(self, actions: list) -> tuple:
        """
        准备动作数据
        
        Args:
            actions: 动作字典列表
            
        Returns:
            (move_targets, turn_targets, turret_targets, shoot_targets)
        """
        move_list = []
        turn_list = []
        turret_list = []
        shoot_list = []
        
        for action in actions:
            move_list.append(action['move'])
            turn_list.append(action['turn'])
            turret_list.append(action['turret'])
            shoot_list.append(action['shoot'])
            
        move_targets = torch.tensor(move_list, dtype=torch.long, device=self.device_)
        turn_targets = torch.tensor(turn_list, dtype=torch.long, device=self.device_)
        turret_targets = torch.tensor(turret_list, dtype=torch.long, device=self.device_)
        shoot_targets = torch.tensor(shoot_list, dtype=torch.long, device=self.device_)
        
        return move_targets, turn_targets, turret_targets, shoot_targets
        
    def saveModel(self, filename: str):
        """保存模型"""
        checkpoint_dir = self.config_.get('training', {}).get('checkpoint_dir', './models/checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filepath = os.path.join(checkpoint_dir, filename)
        
        torch.save({
            'model_state_dict': self.model_.state_dict(),
            'optimizer_state_dict': self.optimizer_.state_dict(),
            'scheduler_state_dict': self.scheduler_.state_dict(),
            'global_step': self.global_step_,
            'best_val_loss': self.best_val_loss_,
            'config': self.config_
        }, filepath)
        
        logger.debug(f"Model saved to {filepath}")
        
    def loadModel(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device_)
        
        self.model_.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler_.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step_ = checkpoint['global_step']
        self.best_val_loss_ = checkpoint['best_val_loss']
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train WoT AI using Imitation Learning")
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/ppo_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../wot_client/data/recordings",
        help="Path to recorded data directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation set split ratio"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}
        
    # 添加模仿学习配置
    if 'imitation' not in config:
        config['imitation'] = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'batch_size': args.batch_size
        }
        
    # 创建数据集
    logger.info(f"Loading dataset from: {args.data_dir}")
    action_mapper = ActionMapper(config)
    
    full_dataset = WotRecordingDataset(
        data_dir=args.data_dir,
        action_mapper=action_mapper,
        image_size=(256, 256),
        augment=True,
        frame_skip=1
    )
    
    if len(full_dataset) == 0:
        logger.error("Dataset is empty! Please record some gameplay data first.")
        return
        
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建训练器
    trainer = ImitationTrainer(config)
    
    # 恢复训练
    if args.resume:
        trainer.loadModel(args.resume)
        
    # 开始训练
    trainer.train(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()

