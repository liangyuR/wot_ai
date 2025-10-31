"""
评估指标模块
"""

import torch
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_action_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                           key_threshold: float = 0.5) -> Dict[str, float]:
    """
    计算动作预测准确率
    
    Args:
        predictions: [B, 11] 预测值
        targets: [B, 11] 真实值
        key_threshold: 按键预测的阈值
        
    Returns:
        包含各种准确率的字典
    """
    with torch.no_grad():
        # 按键部分（前9维）：使用sigmoid + threshold
        key_pred = torch.sigmoid(predictions[:, :9])
        key_target = targets[:, :9]
        key_pred_binary = (key_pred > key_threshold).float()
        key_accuracy = (key_pred_binary == key_target).float().mean().item()
        
        # 鼠标位移部分（后2维）：使用MSE误差
        mouse_pred = predictions[:, 9:11]
        mouse_target = targets[:, 9:11]
        mouse_mse = torch.nn.functional.mse_loss(mouse_pred, mouse_target).item()
        mouse_mae = torch.nn.functional.l1_loss(mouse_pred, mouse_target).item()
        
        # 整体准确率
        overall_accuracy = key_accuracy * 0.7 + (1.0 - min(mouse_mae / 2.0, 1.0)) * 0.3
    
    return {
        'key_accuracy': key_accuracy,
        'mouse_mse': mouse_mse,
        'mouse_mae': mouse_mae,
        'overall_accuracy': overall_accuracy
    }


class ActionMetrics:
    """动作评估指标收集器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.key_accuracies = []
        self.mouse_mses = []
        self.mouse_maes = []
        self.overall_accuracies = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """更新指标"""
        metrics = compute_action_accuracy(predictions, targets)
        self.key_accuracies.append(metrics['key_accuracy'])
        self.mouse_mses.append(metrics['mouse_mse'])
        self.mouse_maes.append(metrics['mouse_mae'])
        self.overall_accuracies.append(metrics['overall_accuracy'])
    
    def compute(self) -> Dict[str, float]:
        """计算平均指标"""
        if not self.key_accuracies:
            return {}
        
        return {
            'avg_key_accuracy': np.mean(self.key_accuracies),
            'avg_mouse_mse': np.mean(self.mouse_mses),
            'avg_mouse_mae': np.mean(self.mouse_maes),
            'avg_overall_accuracy': np.mean(self.overall_accuracies)
        }

