"""
配置管理模块
从YAML文件加载配置，支持默认值和环境变量覆盖
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def LoadConfig(config_path: Optional[Path] = None) -> Dict:
    """加载配置文件"""
    if config_path is None:
        script_dir = Path(__file__).resolve().parent.parent
        config_path = script_dir / "train_config.yaml"
    
    default_config = {
        'image': {'target_width': 256, 'target_height': 256},
        'action': {
            'movement_keys': ['w', 'a', 's', 'd'],
            'mouse_keys': ['mouse_left', 'mouse_right'],
            'special_keys': ['space', 'shift', 't'],
            'speed_keys': ['r', 'f'],
            'mouse_wheel_keys': ['mouse_wheel_up', 'mouse_wheel_down'],
            'action_dim': 15
        },
        'model': {
            'backbone': 'resnet18',  # 可选: resnet18, efficientnet_v2_s, convnext_tiny
            'sequence_length': 4,
            'use_pretrained': True,
            'freeze_backbone': False,
            'temporal_model': 'lstm',  # 可选: lstm, transformer, multihead_attention
            'lstm_hidden_size': 256,
            'lstm_num_layers': 2,
            'transformer_dim': 256,
            'transformer_heads': 8,
            'transformer_layers': 2,
            'use_flash_attention': False,
            'torch_compile': False,  # PyTorch 2.0+ JIT编译
        },
        'training': {
            'optimizer': 'adamw',  # 可选: adam, adamw, lion, adan
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'epochs': 20,
            'num_workers': 12,
            'val_split': 0.2,
            'gradient_accumulation_steps': 1,
            'scheduler': 'cosine_warmup',  # 可选: reduce_on_plateau, cosine_warmup, onecycle
            'warmup_epochs': 2,
            'use_mixed_precision': True,
        },
        'data': {
            'use_opencv': True,  # 使用OpenCV替代PIL
            'enable_cache': True,  # 启用图像缓存
            'cache_size': 1000,  # LRU缓存大小
            'prefetch_factor': 2,  # DataLoader预取因子
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
            # 深度合并默认配置
            config = _deep_merge(default_config, config)
            logger.info(f"已加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return default_config
    else:
        logger.info(f"配置文件不存在: {config_path}，使用默认配置")
        return default_config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并配置字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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

