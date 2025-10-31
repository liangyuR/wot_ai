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
from torchvision import models
from pathlib import Path
from PIL import Image
import json
import os
import yaml
import numpy as np
from typing import List, Tuple, Dict, Optional
import contextlib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ 配置加载 ============
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_config_file = _script_dir / "train_config.yaml"


def LoadConfig(config_path: Optional[Path] = None) -> Dict:
    """加载配置文件"""
    if config_path is None:
        config_path = _config_file
    
    default_config = {
        'image': {'target_width': 256, 'target_height': 256},
        'action': {
            'movement_keys': ['w', 'a', 's', 'd'],
            'mouse_keys': ['mouse_left', 'mouse_right'],
            'special_keys': ['space', 'shift', 't'],
            'action_dim': 11
        },
        'model': {
            'sequence_length': 4,
            'use_pretrained': True,
            'freeze_backbone': False,
            'lstm_hidden_size': 256,
            'lstm_num_layers': 2
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 20,
            'num_workers': 12,
            'val_split': 0.2
        },
        'paths': {
            'data_root': 'C:/data/recordings',
            'model_save_path': 'tank_imitation_model.pth'
        },
        'resolution': {
            'default_width': 3840,
            'default_height': 2160
        }
    }
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            # 合并默认配置
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
                elif isinstance(default_config[key], dict):
                    for subkey in default_config[key]:
                        if subkey not in config[key]:
                            config[key][subkey] = default_config[key][subkey]
            logger.info(f"已加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return default_config
    else:
        logger.info(f"配置文件不存在: {config_path}，使用默认配置")
        return default_config


def LoadResolutionFromData(data_root: str, config: Dict) -> Tuple[int, int]:
    """从数据的meta.json中读取分辨率"""
    data_path = Path(data_root)
    default_w = config['resolution']['default_width']
    default_h = config['resolution']['default_height']
    
    # 查找第一个session的meta.json
    for session_dir in data_path.iterdir():
        if not session_dir.is_dir() or not session_dir.name.startswith("session_"):
            continue
        
        meta_path = session_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                resolution = meta.get('resolution', [default_w, default_h])
                if isinstance(resolution, list) and len(resolution) >= 2:
                    width, height = int(resolution[0]), int(resolution[1])
                    logger.info(f"从 {meta_path} 读取分辨率: {width}x{height}")
                    return width, height
            except Exception as e:
                logger.warning(f"读取meta.json失败: {e}")
                break
    
    logger.info(f"未找到meta.json，使用默认分辨率: {default_w}x{default_h}")
    return default_w, default_h


# 加载配置
_config = LoadConfig()

# 从配置中提取参数
TARGET_W = _config['image']['target_width']
TARGET_H = _config['image']['target_height']

ACTION_KEYS = _config['action']['movement_keys']
MOUSE_KEYS = _config['action']['mouse_keys']
SPECIAL_KEYS = _config['action']['special_keys']
ACTION_DIM = _config['action']['action_dim']

SEQUENCE_LENGTH = _config['model']['sequence_length']
USE_PRETRAINED = _config['model']['use_pretrained']
FREEZE_BACKBONE = _config['model']['freeze_backbone']
LSTM_HIDDEN_SIZE = _config['model']['lstm_hidden_size']
LSTM_NUM_LAYERS = _config['model']['lstm_num_layers']

BATCH_SIZE = _config['training']['batch_size']
LEARNING_RATE = _config['training']['learning_rate']
EPOCHS = _config['training']['epochs']
NUM_WORKERS = _config['training']['num_workers']
VAL_SPLIT = _config['training']['val_split']

# 数据路径（支持环境变量覆盖）
DATA_ROOT = os.environ.get("TRAIN_DATA_ROOT") or _config['paths']['data_root']
MODEL_SAVE_PATH = str(_script_dir / _config['paths']['model_save_path'])

# 分辨率（将在数据集初始化时从meta.json读取）
SCREEN_W, SCREEN_H = LoadResolutionFromData(DATA_ROOT, _config)

# ================================

class TankImitationDataset(Dataset):
    """坦克世界模仿学习数据集"""
    
    def __init__(self, data_root: str, transform=None, sequence_length=SEQUENCE_LENGTH,
                 screen_width: Optional[int] = None, screen_height: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            transform: 图像变换
            sequence_length: 连续帧序列长度
            screen_width: 屏幕宽度（如果为None，将从meta.json读取）
            screen_height: 屏幕高度（如果为None，将从meta.json读取）
        """
        self.data_root = Path(data_root)
        self.samples = []  # (frame_path, action, dx, dy, session_dir, frame_num)
        self.transform = transform or self._get_default_transform()
        self.sequence_length = sequence_length
        
        # 从meta.json读取分辨率（如果未提供）
        if screen_width is None or screen_height is None:
            self.screen_width, self.screen_height = self._load_resolution_from_data()
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height
        
        logger.info(f"使用分辨率: {self.screen_width}x{self.screen_height}")
        
        # 收集所有session的数据
        self._collect_samples()
        logger.info(f"数据集加载完成，共 {len(self.samples)} 个样本")
    
    def _load_resolution_from_data(self) -> Tuple[int, int]:
        """从数据的meta.json中读取分辨率"""
        default_w = SCREEN_W
        default_h = SCREEN_H
        
        # 查找第一个session的meta.json
        for session_dir in self.data_root.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith("session_"):
                continue
            
            meta_path = session_dir / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    resolution = meta.get('resolution', [default_w, default_h])
                    if isinstance(resolution, list) and len(resolution) >= 2:
                        width, height = int(resolution[0]), int(resolution[1])
                        logger.info(f"从 {meta_path} 读取分辨率: {width}x{height}")
                        return width, height
                except Exception as e:
                    logger.debug(f"读取meta.json失败: {e}")
                    break
        
        logger.info(f"未找到meta.json，使用默认分辨率: {default_w}x{default_h}")
        return default_w, default_h
    
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
                    cur_mouse = action.get('mouse_pos', [self.screen_width//2, self.screen_height//2])
                    if last_mouse is None:
                        dx, dy = 0.0, 0.0
                    else:
                        dx = float(cur_mouse[0] - last_mouse[0])
                        dy = float(cur_mouse[1] - last_mouse[1])
                    self.samples.append((frame_path_6, action, dx, dy, session_dir, frame_num))
                    last_mouse = cur_mouse
                    continue

                # 回退尝试4位零填充
                frame_path_4 = frames_dir / f"frame_{frame_num:04d}.jpg"
                if frame_path_4.exists():
                    cur_mouse = action.get('mouse_pos', [self.screen_width//2, self.screen_height//2])
                    if last_mouse is None:
                        dx, dy = 0.0, 0.0
                    else:
                        dx = float(cur_mouse[0] - last_mouse[0])
                        dy = float(cur_mouse[1] - last_mouse[1])
                    self.samples.append((frame_path_4, action, dx, dy, session_dir, frame_num))
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
                        cur_mouse = action.get('mouse_pos', [self.screen_width//2, self.screen_height//2])
                        if last_mouse is None:
                            dx, dy = 0.0, 0.0
                        else:
                            dx = float(cur_mouse[0] - last_mouse[0])
                            dy = float(cur_mouse[1] - last_mouse[1])
                        self.samples.append((cand_path, action, dx, dy, session_dir, frame_num))
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
        """获取连续帧序列样本"""
        frame_path, action, dx, dy, session_dir, frame_num = self.samples[idx]
        
        # 收集连续帧序列
        frame_sequence = []
        action_sequence = []
        
        # 从当前帧向前取sequence_length帧
        for offset in range(-(self.sequence_length - 1), 1):
            target_frame_num = frame_num + offset
            target_frame_path = self._find_frame_path(session_dir, target_frame_num)
            
            if target_frame_path and target_frame_path.exists():
                try:
                    image = Image.open(target_frame_path).convert('RGB')
                    image = self.transform(image)
                    frame_sequence.append(image)
                except Exception as e:
                    logger.debug(f"加载图像失败 {target_frame_path}: {e}")
                    # 如果加载失败，使用当前帧填充
                    if frame_sequence:
                        frame_sequence.append(frame_sequence[-1].clone())
                    else:
                        frame_sequence.append(torch.zeros(3, TARGET_H, TARGET_W))
                
                # 获取对应的动作
                if offset == 0:
                    # 当前帧使用原始动作
                    action_vector = self._encode_action(action, dx, dy)
                else:
                    # 从samples中找到对应帧的动作
                    action_vector = self._get_action_for_frame(session_dir, target_frame_num)
                
                action_sequence.append(action_vector)
            else:
                # 如果找不到帧，用当前帧填充
                if frame_sequence:
                    frame_sequence.append(frame_sequence[-1].clone())
                    action_sequence.append(action_sequence[-1].clone())
                else:
                    # 如果当前帧都找不到，返回零填充
                    frame_sequence.append(torch.zeros(3, TARGET_H, TARGET_W))
                    action_sequence.append(torch.zeros(ACTION_DIM))
        
        # 确保序列长度正确（如果不足，用第一帧填充）
        while len(frame_sequence) < self.sequence_length:
            if frame_sequence:
                frame_sequence.insert(0, frame_sequence[0].clone())
                action_sequence.insert(0, action_sequence[0].clone())
            else:
                frame_sequence.insert(0, torch.zeros(3, TARGET_H, TARGET_W))
                action_sequence.insert(0, torch.zeros(ACTION_DIM))
        
        # 堆叠为张量
        frames_tensor = torch.stack(frame_sequence[:self.sequence_length])  # [T, 3, H, W]
        target_action = action_sequence[-1]  # 使用最后一帧的动作作为目标
        
        return frames_tensor, target_action
    
    def _extract_frame_number(self, frame_path: Path) -> int:
        """从文件路径提取帧号"""
        filename = frame_path.name
        # 支持 frame_000576.jpg 或 frame_0576.jpg
        try:
            if filename.startswith('frame_') and filename.endswith('.jpg'):
                num_str = filename[6:-4]  # 去掉 'frame_' 和 '.jpg'
                return int(num_str)
        except ValueError:
            pass
        return 0
    
    def _find_frame_path(self, session_dir: Path, frame_num: int) -> Path:
        """查找指定帧号的文件路径"""
        frames_dir = session_dir / "frames"
        
        # 尝试6位零填充
        frame_path_6 = frames_dir / f"frame_{frame_num:06d}.jpg"
        if frame_path_6.exists():
            return frame_path_6
        
        # 尝试4位零填充
        frame_path_4 = frames_dir / f"frame_{frame_num:04d}.jpg"
        if frame_path_4.exists():
            return frame_path_4
        
        # 尝试就近偶数帧
        nearest_even = (frame_num // 2) * 2
        candidates = [nearest_even, nearest_even - 2, nearest_even + 2]
        for cand in candidates:
            if cand < 0:
                continue
            cand_path = frames_dir / f"frame_{cand:06d}.jpg"
            if cand_path.exists():
                return cand_path
        
        return None
    
    def _get_action_for_frame(self, session_dir: Path, frame_num: int) -> torch.Tensor:
        """获取指定帧号对应的动作向量"""
        # 在samples中查找相同session和帧号的样本
        for sample in self.samples:
            sample_frame_path, sample_action, sample_dx, sample_dy, sample_session, sample_frame_num = sample
            if sample_session == session_dir and sample_frame_num == frame_num:
                return self._encode_action(sample_action, sample_dx, sample_dy)
        
        # 如果找不到，返回零向量
        return torch.zeros(ACTION_DIM, dtype=torch.float32)
    
    def _encode_action(self, action: Dict, dx: float, dy: float) -> torch.Tensor:
        """
        将动作编码为向量（使用鼠标相对位移）
        
        输出格式: [W, A, S, D, mouse_left, mouse_right, dx_norm, dy_norm, space, shift, t]
        共11维
        """
        keys = action.get('keys', [])
        
        # 编码移动按键 [0:4]
        key_states = []
        for key in ACTION_KEYS:
            key_states.append(1.0 if key in keys else 0.0)
        
        # 编码鼠标按键 [4:6]
        mouse_left = 1.0 if "mouse_left" in keys else 0.0
        mouse_right = 1.0 if "mouse_right" in keys else 0.0
        
        # 归一化相对位移到 [-1, 1] [6:8]
        max_step = 50.0
        dx_norm = float(np.clip(dx / max_step, -1.0, 1.0))
        dy_norm = float(np.clip(dy / max_step, -1.0, 1.0))
        
        # 编码特殊按键 [8:11]
        special_states = []
        for key in SPECIAL_KEYS:
            special_states.append(1.0 if key in keys else 0.0)
        
        # 组合动作向量
        action_vector = (key_states + [mouse_left, mouse_right, dx_norm, dy_norm] + 
                        special_states)
        
        return torch.tensor(action_vector, dtype=torch.float32)


class TankImitationNet(nn.Module):
    """坦克世界模仿学习网络 - ResNet18 + LSTM时序模型"""
    
    def __init__(self, action_dim: int = ACTION_DIM, sequence_length: int = SEQUENCE_LENGTH,
                 use_pretrained: bool = USE_PRETRAINED, freeze_backbone: bool = FREEZE_BACKBONE):
        """
        初始化网络
        
        Args:
            action_dim: 动作维度 (默认11: W, A, S, D, mouse_left, mouse_right, dx, dy, space, shift, t)
            sequence_length: 连续帧序列长度
            use_pretrained: 是否使用ImageNet预训练权重
            freeze_backbone: 是否冻结ResNet18 backbone
        """
        super(TankImitationNet, self).__init__()
        
        # ResNet18特征提取器（ImageNet预训练）
        resnet = models.resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)
        # 移除最后的全连接层，保留特征提取部分
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 冻结backbone（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("ResNet18 backbone已冻结")
        
        # ResNet18输出特征维度
        self.feature_dim = 512  # resnet18的最后一层是512维
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=0.2 if LSTM_NUM_LAYERS > 1 else 0
        )
        
        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, action_dim)
        )
        
        self.sequence_length = sequence_length
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, 3, H, W] 输入帧序列
            
        Returns:
            [B, action_dim] 动作预测
        """
        B, T, C, H, W = x.shape
        
        # 将所有帧展平，通过ResNet18提取特征
        x_flat = x.view(B * T, C, H, W)  # [B*T, 3, H, W]
        features = self.backbone(x_flat)  # [B*T, 512, 1, 1]
        features = features.view(B * T, -1)  # [B*T, 512]
        features = features.view(B, T, -1)  # [B, T, 512]
        
        # LSTM时序建模
        lstm_out, (h_n, c_n) = self.lstm(features)  # [B, T, LSTM_HIDDEN_SIZE]
        
        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]  # [B, LSTM_HIDDEN_SIZE]
        
        # 动作预测
        action = self.action_head(last_hidden)  # [B, action_dim]
        
        return action


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
        
        # 损失函数 - 对按键使用BCE，对鼠标位移使用MSE
        # 动作向量格式: [W, A, S, D, mouse_left, mouse_right, dx_norm, dy_norm, space, shift, t]
        # 前9维是按键（二进制），后2维是鼠标位移（连续值）
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
            
            # 分离按键和鼠标位移
            # 按键部分：前9维 [W, A, S, D, mouse_left, mouse_right, space, shift, t]
            key_pred = predictions[:, :9]
            # 鼠标位移部分：后2维 [dx_norm, dy_norm]
            mouse_pred = predictions[:, 9:11]
            
            key_target = actions[:, :9]
            mouse_target = actions[:, 9:11]
            
            # 计算损失（新AMP API）
            with amp_ctx:
                key_loss = self.key_loss_fn(key_pred, key_target)
                mouse_loss = self.mouse_loss_fn(mouse_pred, mouse_target)
                # 加权组合：按键损失权重1.0，鼠标位移损失权重0.5
                total_loss_batch = key_loss + 0.5 * mouse_loss

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


def create_data_loaders(data_root, batch_size=BATCH_SIZE, val_split=VAL_SPLIT):
    """创建数据加载器"""
    # 创建数据集（分辨率将从meta.json自动读取）
    dataset = TankImitationDataset(data_root, sequence_length=SEQUENCE_LENGTH)
    
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
    model = TankImitationNet(
        action_dim=ACTION_DIM,
        sequence_length=SEQUENCE_LENGTH,
        use_pretrained=USE_PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE
    )
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建训练器
    trainer = ImitationTrainer(model, train_loader, val_loader, device)
    
    # 开始训练
    trainer.train(EPOCHS)
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
