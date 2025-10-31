"""
常量定义模块
从配置中提取常量，供其他模块使用
"""

import os
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# 延迟加载配置，避免循环导入
_config: Dict = None
_initialized = False


def _init_config():
    """初始化配置（延迟加载）"""
    global _config, _initialized
    if _initialized:
        return
    
    from .config import LoadConfig, LoadResolutionFromData
    
    script_dir = Path(__file__).resolve().parent.parent
    _config = LoadConfig()
    
    # 提取配置常量
    global TARGET_W, TARGET_H
    global ACTION_KEYS, MOUSE_KEYS, SPECIAL_KEYS, ACTION_DIM
    global SEQUENCE_LENGTH, USE_PRETRAINED, FREEZE_BACKBONE
    global LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS
    global BATCH_SIZE, LEARNING_RATE, EPOCHS, NUM_WORKERS, VAL_SPLIT
    global DATA_ROOT, MODEL_SAVE_PATH, SCREEN_W, SCREEN_H
    global BACKBONE_TYPE, TEMPORAL_MODEL, OPTIMIZER_TYPE
    global USE_OPENCV, ENABLE_CACHE, CACHE_SIZE, PREFETCH_FACTOR
    global USE_TORCH_COMPILE
    
    TARGET_W = _config['image']['target_width']
    TARGET_H = _config['image']['target_height']
    
    ACTION_KEYS = _config['action']['movement_keys']
    MOUSE_KEYS = _config['action']['mouse_keys']
    SPECIAL_KEYS = _config['action']['special_keys']
    ACTION_DIM = _config['action']['action_dim']
    
    SEQUENCE_LENGTH = _config['model']['sequence_length']
    USE_PRETRAINED = _config['model']['use_pretrained']
    FREEZE_BACKBONE = _config['model']['freeze_backbone']
    BACKBONE_TYPE = _config['model']['backbone']
    TEMPORAL_MODEL = _config['model']['temporal_model']
    LSTM_HIDDEN_SIZE = _config['model']['lstm_hidden_size']
    LSTM_NUM_LAYERS = _config['model']['lstm_num_layers']
    USE_TORCH_COMPILE = _config['model'].get('torch_compile', False)
    
    BATCH_SIZE = _config['training']['batch_size']
    LEARNING_RATE = _config['training']['learning_rate']
    EPOCHS = _config['training']['epochs']
    NUM_WORKERS = _config['training']['num_workers']
    VAL_SPLIT = _config['training']['val_split']
    OPTIMIZER_TYPE = _config['training']['optimizer']
    
    USE_OPENCV = _config['data'].get('use_opencv', True)
    ENABLE_CACHE = _config['data'].get('enable_cache', True)
    CACHE_SIZE = _config['data'].get('cache_size', 1000)
    PREFETCH_FACTOR = _config['data'].get('prefetch_factor', 2)
    
    # 数据路径（支持环境变量覆盖）
    DATA_ROOT = os.environ.get("TRAIN_DATA_ROOT") or _config['paths']['data_root']
    MODEL_SAVE_PATH = str(script_dir / _config['paths']['model_save_path'])
    
    # 分辨率（将在数据集初始化时从meta.json读取）
    SCREEN_W, SCREEN_H = LoadResolutionFromData(DATA_ROOT, _config)
    
    _initialized = True


def GetConfig() -> Dict:
    """获取完整配置字典"""
    _init_config()
    return _config


# 初始化常量
_init_config()

