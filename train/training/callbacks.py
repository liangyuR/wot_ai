"""
训练回调模块
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
import torch

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """训练回调基类"""
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, **kwargs):
        """epoch开始时的回调"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """epoch结束时的回调"""
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """batch结束时的回调（可选）"""
        pass


class ModelCheckpoint(TrainingCallback):
    """模型检查点回调"""
    
    def __init__(self, save_path: str, monitor: str = 'val_loss', mode: str = 'min', 
                 save_best: bool = True, save_last: bool = True):
        """
        Args:
            save_path: 保存路径前缀
            monitor: 监控指标
            mode: 'min' 或 'max'
            save_best: 是否保存最佳模型
            save_last: 是否保存最后一个模型
        """
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
    
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model=None, 
                     optimizer=None, **kwargs):
        if model is None:
            return
        
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return
        
        # 检查是否是最佳值
        is_best = False
        if self.mode == 'min':
            if current_value < self.best_value:
                self.best_value = current_value
                self.best_epoch = epoch
                is_best = True
        else:
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_epoch = epoch
                is_best = True
        
        # 保存最佳模型
        if is_best and self.save_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'metrics': metrics,
            }
            best_path = f"{self.save_path}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型 (epoch {epoch}, {self.monitor}={current_value:.4f}): {best_path}")
        
        # 保存最后一个模型
        if self.save_last:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'metrics': metrics,
            }
            last_path = f"{self.save_path}_last.pth"
            torch.save(checkpoint, last_path)
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        pass


class LoggingCallback(TrainingCallback):
    """日志回调"""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
    
    def on_epoch_start(self, epoch: int, **kwargs):
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch} 开始")
        logger.info(f"{'='*60}")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        logger.info(f"Epoch {epoch} 完成")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        if batch_idx % self.log_interval == 0:
            logger.info(f"  Batch {batch_idx}, Loss: {loss:.4f}")

