#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克世界行为克隆训练脚本
基于收集的游戏数据进行端到端模仿学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path
from PIL import Image
import json
import os
import numpy as np
from typing import List, Tuple, Dict
import contextlib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ 配置参数 ============
SCREEN_W, SCREEN_H = 3840, 2160  # 原始分辨率
TARGET_W, TARGET_H = 256, 256     # 模型输入分辨率
ACTION_KEYS = ["w", "a", "s", "d"]  # 移动按键
MOUSE_KEYS = ["mouse_left", "mouse_right"]  # 鼠标按键
BATCH_SIZE = 96
LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_WORKERS = 12

# 动态计算数据路径（相对于脚本位置）
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
DATA_ROOT = "C:/data/recordings"
MODEL_SAVE_PATH = str(_script_dir / "tank_imitation_model.pth")
# ================================

class TankImitationDataset(Dataset):
    """坦克世界模仿学习数据集"""
    
    def __init__(self, data_root: str, transform=None):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            transform: 图像变换
        """
        self.data_root = Path(data_root)
        self.samples = []  # (frame_path, action, dx, dy)
        self.transform = transform or self._get_default_transform()
        
        # 收集所有session的数据
        self._collect_samples()
        logger.info(f"数据集加载完成，共 {len(self.samples)} 个样本")
    
    def _get_default_transform(self):
        """获取默认的图像变换"""
        return transforms.Compose([
            transforms.Resize((TARGET_H, TARGET_W), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _collect_samples(self):
        """收集所有session的样本，并为鼠标计算相对位移(dx, dy)"""
        for session_dir in self.data_root.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith("session_"):
                continue
                
            actions_file = session_dir / "actions.json"
            frames_dir = session_dir / "frames"
            
            if not actions_file.exists() or not frames_dir.exists():
                logger.warning(f"跳过不完整的session: {session_dir.name}")
                continue
            
            # 加载动作数据
            try:
                with open(actions_file, 'r', encoding='utf-8') as f:
                    actions = json.load(f)
            except Exception as e:
                logger.error(f"加载动作文件失败 {actions_file}: {e}")
                continue
            
            # 收集有效样本，计算相对位移
            last_mouse = None
            for action in actions:
                frame_num = action.get('frame', 0)
                # 适配6位零填充文件名（数据实际为6位，例如 frame_000576.jpg）
                frame_path_6 = frames_dir / f"frame_{frame_num:06d}.jpg"
                if frame_path_6.exists():
                    cur_mouse = action.get('mouse_pos', [SCREEN_W//2, SCREEN_H//2])
                    if last_mouse is None:
                        dx, dy = 0.0, 0.0
                    else:
                        dx = float(cur_mouse[0] - last_mouse[0])
                        dy = float(cur_mouse[1] - last_mouse[1])
                    self.samples.append((frame_path_6, action, dx, dy))
                    last_mouse = cur_mouse
                    continue

                # 回退尝试4位零填充
                frame_path_4 = frames_dir / f"frame_{frame_num:04d}.jpg"
                if frame_path_4.exists():
                    cur_mouse = action.get('mouse_pos', [SCREEN_W//2, SCREEN_H//2])
                    if last_mouse is None:
                        dx, dy = 0.0, 0.0
                    else:
                        dx = float(cur_mouse[0] - last_mouse[0])
                        dy = float(cur_mouse[1] - last_mouse[1])
                    self.samples.append((frame_path_4, action, dx, dy))
                    last_mouse = cur_mouse
                    continue

                # 如果未找到，依据可能的 frame_step（例如 step=2，仅保存偶数帧）尝试就近偶数帧
                nearest_even = (frame_num // 2) * 2
                candidates = [nearest_even, nearest_even - 2, nearest_even + 2]
                found = False
                for cand in candidates:
                    if cand < 0:
                        continue
                    cand_path = frames_dir / f"frame_{cand:06d}.jpg"
                    if cand_path.exists():
                        cur_mouse = action.get('mouse_pos', [SCREEN_W//2, SCREEN_H//2])
                        if last_mouse is None:
                            dx, dy = 0.0, 0.0
                        else:
                            dx = float(cur_mouse[0] - last_mouse[0])
                            dy = float(cur_mouse[1] - last_mouse[1])
                        self.samples.append((cand_path, action, dx, dy))
                        last_mouse = cur_mouse
                        found = True
                        break
                if found:
                    continue

            if not self.samples:
                logger.warning(f"未在 {session_dir} 收集到任何样本，请检查帧文件命名与actions对齐情况")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        frame_path, action, dx, dy = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(frame_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.error(f"加载图像失败 {frame_path}: {e}")
            # 返回零图像
            image = torch.zeros(3, TARGET_H, TARGET_W)
        
        # 编码动作向量（包含相对鼠标位移）
        action_vector = self._encode_action(action, dx, dy)
        
        return image, action_vector
    
    def _encode_action(self, action: Dict, dx: float, dy: float) -> torch.Tensor:
        """
        将动作编码为向量（使用鼠标相对位移）
        
        输出格式: [W, A, S, D, mouse_left, mouse_right, dx_norm, dy_norm]
        """
        keys = action.get('keys', [])
        
        # 编码按键状态
        key_states = []
        for key in ACTION_KEYS:
            key_states.append(1.0 if key in keys else 0.0)
        
        # 编码鼠标按键
        mouse_left = 1.0 if "mouse_left" in keys else 0.0
        mouse_right = 1.0 if "mouse_right" in keys else 0.0
        
        # 归一化相对位移到 [-1, 1]
        max_step = 50.0
        dx_norm = float(np.clip(dx / max_step, -1.0, 1.0))
        dy_norm = float(np.clip(dy / max_step, -1.0, 1.0))
        
        # 组合动作向量
        action_vector = key_states + [mouse_left, mouse_right, dx_norm, dy_norm]
        
        return torch.tensor(action_vector, dtype=torch.float32)


class TankImitationNet(nn.Module):
    """坦克世界模仿学习网络"""
    
    def __init__(self, action_dim: int = 8):
        """
        初始化网络
        
        Args:
            action_dim: 动作维度 (W, A, S, D, mouse_left, mouse_right, mouse_x, mouse_y)
        """
        super(TankImitationNet, self).__init__()
        
        # CNN特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 计算卷积输出尺寸
        conv_output_size = self._get_conv_output_size()
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, action_dim)
        )
    
    def _get_conv_output_size(self):
        """计算卷积层输出尺寸"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, TARGET_H, TARGET_W)
            dummy_output = self.feature_extractor(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def forward(self, x):
        """前向传播"""
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class ImitationTrainer:
    """模仿学习训练器"""
    
    def __init__(self, model, train_loader, val_loader=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_cuda = (device.type == 'cuda')
        
        # 性能优化设置
        if self.use_cuda:
            # 使用新API控制TF32精度，替代已弃用的 allow_tf32
            try:
                # 矩阵乘法与卷积的FP32精度策略
                # matmul 受 set_float32_matmul_precision 及 fp32_precision 共同影响
                torch.backends.cuda.matmul.fp32_precision = 'tf32'  # 可选: 'ieee' 禁用TF32
            except Exception:
                pass
            try:
                torch.backends.cudnn.conv.fp32_precision = 'tf32'
            except Exception:
                pass

        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # 损失函数 - 对按键使用BCE，对鼠标位置使用MSE
        self.key_loss_fn = nn.BCEWithLogitsLoss()
        self.mouse_loss_fn = nn.MSELoss()
        
        # AMP 混合精度（新API）
        self.scaler = torch.amp.GradScaler('cuda') if self.use_cuda else None

        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """训练一个epoch"""
        # 模型到channels_last以提升GPU卷积效率
        if self.use_cuda:
            self.model.to(memory_format=torch.channels_last)
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, actions) in enumerate(self.train_loader):
            if self.use_cuda:
                images = images.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                actions = actions.to(self.device, non_blocking=True)
            else:
                images = images.to(self.device)
                actions = actions.to(self.device)
            
            # 前向传播（新AMP API）
            amp_ctx = torch.amp.autocast('cuda') if self.use_cuda else contextlib.nullcontext()
            with amp_ctx:
                predictions = self.model(images)
            
            # 分离按键和鼠标位置
            key_pred = predictions[:, :6]  # W, A, S, D, mouse_left, mouse_right
            mouse_pred = predictions[:, 6:]  # mouse_x, mouse_y
            
            key_target = actions[:, :6]
            mouse_target = actions[:, 6:]
            
            # 计算损失（新AMP API）
            with amp_ctx:
                key_loss = self.key_loss_fn(key_pred, key_target)
                mouse_loss = self.mouse_loss_fn(mouse_pred, mouse_target)
                total_loss_batch = key_loss + mouse_loss

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {total_loss_batch.item():.4f}, '
                          f'Key Loss: {key_loss.item():.4f}, '
                          f'Mouse Loss: {mouse_loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """验证模型"""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, actions in self.val_loader:
                if self.use_cuda:
                    images = images.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                    actions = actions.to(self.device, non_blocking=True)
                else:
                    images = images.to(self.device)
                    actions = actions.to(self.device)

                amp_ctx = torch.amp.autocast('cuda') if self.use_cuda else contextlib.nullcontext()
                with amp_ctx:
                    predictions = self.model(images)
                
                # 分离按键和鼠标位置
                key_pred = predictions[:, :6]
                mouse_pred = predictions[:, 6:]
                
                key_target = actions[:, :6]
                mouse_target = actions[:, 6:]
                
                # 计算损失
                with amp_ctx:
                    key_loss = self.key_loss_fn(key_pred, key_target)
                    mouse_loss = self.mouse_loss_fn(mouse_pred, mouse_target)
                    total_loss_batch = key_loss + mouse_loss
                total_loss += total_loss_batch.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, epochs):
        """完整训练流程"""
        logger.info(f"开始训练，共 {epochs} 个epoch")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            logger.info(f"训练损失: {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate()
            if val_loss is not None:
                logger.info(f"验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"best_model_epoch_{epoch+1}.pth")
                    logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
                
                # 学习率调度（手动记录LR变化）
                prev_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr < prev_lr:
                    logger.info(f"学习率下降: {prev_lr:.3e} -> {new_lr:.3e}")
            else:
                # 没有验证集时保存当前模型
                if epoch % 5 == 0:
                    self.save_model(f"model_epoch_{epoch+1}.pth")
        
        # 保存最终模型
        self.save_model(MODEL_SAVE_PATH)
        logger.info(f"训练完成，模型已保存到 {MODEL_SAVE_PATH}")
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)


def create_data_loaders(data_root, batch_size=BATCH_SIZE, val_split=0.2):
    """创建数据加载器"""
    # 创建数据集
    dataset = TankImitationDataset(data_root)
    
    # 划分训练集和验证集
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoader 性能选项
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
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
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
    model = TankImitationNet(action_dim=8)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = ImitationTrainer(model, train_loader, val_loader, device)
    
    # 开始训练
    trainer.train(EPOCHS)
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
