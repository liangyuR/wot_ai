"""
优化的训练器模块
支持现代优化器、学习率调度、梯度累积、性能监控
"""

import torch
import torch.nn as nn
import contextlib
import time
from typing import Optional, List, Dict
import logging

from ..core.constants import (
    LEARNING_RATE, BATCH_SIZE, OPTIMIZER_TYPE, NUM_WORKERS,
    VAL_SPLIT, USE_TORCH_COMPILE, ACTION_DIM
)

logger = logging.getLogger(__name__)


def create_optimizer(model, optimizer_type: str = 'adamw', lr: float = None, 
                    weight_decay: float = 1e-5, **kwargs):
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型 ('adam', 'adamw', 'lion', 'adan')
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数
        
    Returns:
        优化器
    """
    lr = lr or LEARNING_RATE
    
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'lion':
        try:
            from lion_pytorch import Lion
            optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info("使用Lion优化器")
        except ImportError:
            logger.warning("lion_pytorch未安装，回退到AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adan':
        try:
            from adan import Adan
            optimizer = Adan(model.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info("使用Adan优化器")
        except ImportError:
            logger.warning("adan未安装，回退到AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logger.warning(f"未知优化器类型 {optimizer_type}，使用AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    return optimizer


def create_scheduler(optimizer, scheduler_type: str = 'cosine_warmup', 
                    epochs: int = 20, warmup_epochs: int = 2, **kwargs):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('reduce_on_plateau', 'cosine_warmup', 'onecycle')
        epochs: 总epoch数
        warmup_epochs: warmup轮数
        **kwargs: 其他参数
        
    Returns:
        调度器
    """
    if scheduler_type == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    elif scheduler_type == 'cosine_warmup':
        # CosineAnnealingLR with Warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=kwargs.get('max_lr', LEARNING_RATE * 10),
            total_steps=epochs, pct_start=0.1
        )
    else:
        logger.warning(f"未知调度器类型 {scheduler_type}，使用ReduceLROnPlateau")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
    
    return scheduler


class ImitationTrainer:
    """优化的模仿学习训练器"""
    
    def __init__(self, model, train_loader, val_loader=None, device='cuda',
                 optimizer_type: str = None, scheduler_type: str = None,
                 gradient_accumulation_steps: int = 1,
                 callbacks: Optional[List] = None):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
            optimizer_type: 优化器类型
            scheduler_type: 学习率调度器类型
            gradient_accumulation_steps: 梯度累积步数
            callbacks: 训练回调列表
        """
        from ..core.constants import GetConfig
        config = GetConfig()
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_cuda = (device.type == 'cuda')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.callbacks = callbacks or []
        
        # 性能优化设置
        if self.use_cuda:
            try:
                torch.backends.cuda.matmul.fp32_precision = 'tf32'
            except Exception:
                pass
            try:
                torch.backends.cudnn.conv.fp32_precision = 'tf32'
            except Exception:
                pass
        
        # 创建优化器
        optimizer_type = optimizer_type or OPTIMIZER_TYPE
        weight_decay = config['training'].get('weight_decay', 1e-5)
        self.optimizer = create_optimizer(
            model, optimizer_type=optimizer_type,
            lr=LEARNING_RATE, weight_decay=weight_decay
        )
        
        # 创建学习率调度器
        scheduler_type = scheduler_type or config['training'].get('scheduler', 'cosine_warmup')
        epochs = config['training']['epochs']
        warmup_epochs = config['training'].get('warmup_epochs', 2)
        self.scheduler = create_scheduler(
            self.optimizer, scheduler_type=scheduler_type,
            epochs=epochs, warmup_epochs=warmup_epochs
        )
        
        # 损失函数
        self.key_loss_fn = nn.BCEWithLogitsLoss()
        self.mouse_loss_fn = nn.MSELoss()
        
        # AMP 混合精度
        self.scaler = torch.amp.GradScaler('cuda') if self.use_cuda else None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        # 性能统计
        self.batch_times = []
        self.data_load_times = []
    
    def train_epoch(self, epoch: int = 0):
        """训练一个epoch"""
        # 触发回调
        for callback in self.callbacks:
            callback.on_epoch_start(epoch)
        
        # 模型到channels_last以提升GPU卷积效率
        if self.use_cuda:
            self.model.to(memory_format=torch.channels_last)
        self.model.train()
        
        total_loss = 0.0
        total_key_loss = 0.0
        total_mouse_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, actions) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # 数据加载时间统计
            data_load_time = time.time() - batch_start_time
            self.data_load_times.append(data_load_time)
            
            if self.use_cuda:
                images = images.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                actions = actions.to(self.device, non_blocking=True)
            else:
                images = images.to(self.device)
                actions = actions.to(self.device)
            
            # 前向传播
            amp_ctx = torch.amp.autocast('cuda') if self.use_cuda else contextlib.nullcontext()
            with amp_ctx:
                predictions = self.model(images)
            
            # 分离按键和鼠标位移
            key_pred = predictions[:, :9]
            mouse_pred = predictions[:, 9:11]
            
            key_target = actions[:, :9]
            mouse_target = actions[:, 9:11]
            
            # 计算损失
            with amp_ctx:
                key_loss = self.key_loss_fn(key_pred, key_target)
                mouse_loss = self.mouse_loss_fn(mouse_pred, mouse_target)
                total_loss_batch = key_loss + 0.5 * mouse_loss
            
            # 梯度累积
            if self.scaler is not None:
                self.scaler.scale(total_loss_batch / self.gradient_accumulation_steps).backward()
            else:
                (total_loss_batch / self.gradient_accumulation_steps).backward()
            
            # 更新参数（每gradient_accumulation_steps步更新一次）
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # 更新学习率（OneCycleLR需要每步更新）
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
            
            total_loss += total_loss_batch.item()
            total_key_loss += key_loss.item()
            total_mouse_loss += mouse_loss.item()
            num_batches += 1
            
            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)
            
            # 触发batch回调
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, total_loss_batch.item())
        
        # 最后一个batch的梯度更新
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        avg_loss = total_loss / num_batches
        avg_key_loss = total_key_loss / num_batches
        avg_mouse_loss = total_mouse_loss / num_batches
        
        metrics = {
            'train_loss': avg_loss,
            'train_key_loss': avg_key_loss,
            'train_mouse_loss': avg_mouse_loss,
            'avg_batch_time': sum(self.batch_times[-100:]) / min(100, len(self.batch_times)),
        }
        
        self.train_losses.append(avg_loss)
        
        # 触发回调
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, model=self.model, optimizer=self.optimizer)
        
        return metrics
    
    def validate(self):
        """验证模型"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        total_key_loss = 0.0
        total_mouse_loss = 0.0
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
                
                key_pred = predictions[:, :9]
                mouse_pred = predictions[:, 9:11]
                
                key_target = actions[:, :9]
                mouse_target = actions[:, 9:11]
                
                with amp_ctx:
                    key_loss = self.key_loss_fn(key_pred, key_target)
                    mouse_loss = self.mouse_loss_fn(mouse_pred, mouse_target)
                    total_loss_batch = key_loss + 0.5 * mouse_loss
                
                total_loss += total_loss_batch.item()
                total_key_loss += key_loss.item()
                total_mouse_loss += mouse_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_key_loss = total_key_loss / num_batches
        avg_mouse_loss = total_mouse_loss / num_batches
        
        metrics = {
            'val_loss': avg_loss,
            'val_key_loss': avg_key_loss,
            'val_mouse_loss': avg_mouse_loss,
        }
        
        self.val_losses.append(avg_loss)
        return metrics
    
    def train(self, epochs: int):
        """完整训练流程"""
        from ..core.constants import MODEL_SAVE_PATH
        
        logger.info(f"开始训练，共 {epochs} 个epoch")
        logger.info(f"梯度累积步数: {self.gradient_accumulation_steps}")
        logger.info(f"优化器: {type(self.optimizer).__name__}")
        logger.info(f"学习率调度器: {type(self.scheduler).__name__}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            # 合并指标
            all_metrics = {**train_metrics}
            if val_metrics:
                all_metrics.update(val_metrics)
                val_loss = val_metrics['val_loss']
                
                # 更新学习率（ReduceLROnPlateau需要在验证后更新）
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    # CosineWarmup等其他调度器在epoch结束时更新
                    self.scheduler.step()
                
                # 记录学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"当前学习率: {current_lr:.6f}")
            else:
                # 没有验证集时也更新调度器
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and \
                   not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
        
        # 保存最终模型
        final_path = MODEL_SAVE_PATH
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, final_path)
        logger.info(f"训练完成，模型已保存到 {final_path}")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)

