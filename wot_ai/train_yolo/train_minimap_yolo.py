#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键训练 YOLO 小地图检测模型
自动检查数据集结构、生成 dataset.yaml、启动训练并输出结果
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

# 统一导入机制
from wot_ai.utils.paths import setup_python_path, resolve_path, get_datasets_dir, get_models_dir, get_training_dir, ensure_dir
from wot_ai.utils.imports import import_function
setup_python_path()

# 导入依赖（使用便捷方法）
SetupLogger = import_function([
    'wot_ai.game_modules.common.utils.logger',
    'game_modules.common.utils.logger',
    'common.utils.logger',
    'yolo.utils.logger'
], 'SetupLogger')
if SetupLogger is None:
    from wot_ai.game_modules.common.utils.logger import SetupLogger

PrepareMinimapDataset = import_function([
    'wot_ai.train_yolo.prepare_minimap_dataset',
    'train_yolo.prepare_minimap_dataset'
], 'PrepareMinimapDataset')
if PrepareMinimapDataset is None:
    from .prepare_minimap_dataset import PrepareMinimapDataset

# 从配置文件加载类别定义
from .config_loader import LoadClassesFromConfig

logger = SetupLogger(__name__)


def LoadConfig(config_path: Path) -> dict:
    """
    加载训练配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return GetDefaultConfig()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # 合并默认配置
        default_config = GetDefaultConfig()
        config = _MergeConfig(default_config, config)
        
        # 确保类别从配置文件加载（覆盖默认值）
        if 'classes' in config and isinstance(config['classes'], list):
            # 类别已经在配置文件中，直接使用
            pass
        else:
            # 从配置文件重新加载类别
            config['classes'] = LoadClassesFromConfig(config_path)
        
        # 保存配置文件路径，供后续使用
        config['_config_path'] = str(config_path)
        
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return GetDefaultConfig()


def GetDefaultConfig() -> dict:
    """获取默认配置"""
    # 使用统一路径管理
    datasets_dir = get_datasets_dir()
    training_dir = get_training_dir()
    
    return {
        'model': {
            'name': 'yolo11m.pt',
            'imgsz': 640,
            'device': 'cpu'
        },
        'training': {
            'epochs': 800,
            'batch': 8,
            'workers': 8,
            'augment': True,
            'optimizer': 'AdamW',
            'lr0': 0.002,
            'lrf': 0.01,
            'warmup_epochs': 3.0,
            'patience': 100,
            'dropout': 0.05,
            'loss': {
                'cls': 0.3,
                'box': 7.5,
                'dfl': 1.5
            },
            'mosaic': 1.0,
            'copy_paste': 0.2,
            'mixup': 0.05,
            'close_mosaic': 20
        },
        'augmentation': {
            'hsv_h': 0.02,
            'hsv_s': 0.7,
            'hsv_v': 0.5,
            'degrees': 5,
            'scale': 0.5,
            'translate': 0.15,
            'fliplr': 0.2,
            'brightness': 0.2,
            'contrast': 0.2,
            'blur': 0.1
        },
        'dataset': {
            'source_dir': str(datasets_dir / 'raw' / 'minimap'),
            'output_dir': str(datasets_dir / 'processed' / 'minimap'),
            'train_ratio': 0.8,
            'random_seed': 42
        },
        'classes': LoadClassesFromConfig(),  # 从默认配置文件加载
        'output': {
            'project': str(training_dir / 'minimap'),
            'name': 'exp_minimap',
            'save': True,
            'save_period': 10,
            'val': True
        }
    }


def _MergeConfig(default: dict, override: dict) -> dict:
    """深度合并配置字典"""
    result = default.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _MergeConfig(result[key], value)
        else:
            result[key] = value
    
    return result


def CheckDatasetStructure(dataset_dir: Path) -> bool:
    """
    检查数据集结构是否正确
    
    Args:
        dataset_dir: 数据集目录
    
    Returns:
        是否结构正确
    """
    required_dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.error(f"数据集目录结构不完整，缺少: {dir_path}")
            return False
    
    # 检查是否有文件
    train_images = list((dataset_dir / "images" / "train").glob("*.png"))
    val_images = list((dataset_dir / "images" / "val").glob("*.png"))
    
    if len(train_images) == 0:
        logger.error("训练集图像为空")
        return False
    
    if len(val_images) == 0:
        logger.warning("验证集图像为空，建议添加验证数据")
    
    logger.info(f"数据集结构检查通过: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")
    return True


def GenerateDatasetYaml(dataset_dir: Path, classes: list) -> Path:
    """
    自动生成 dataset.yaml 文件
    
    Args:
        dataset_dir: 数据集目录
        classes: 类别列表
    
    Returns:
        yaml 文件路径
    """
    yaml_path = dataset_dir / "dataset.yaml"
    
    yaml_data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)
    
    logger.info(f"dataset.yaml 已生成: {yaml_path}")
    return yaml_path


def TrainYOLO(config: dict, prepare_dataset: bool = True) -> bool:
    """
    执行 YOLO 模型训练
    
    Args:
        config: 训练配置
        prepare_dataset: 是否自动准备数据集
    
    Returns:
        是否成功
    """
    script_dir = Path(__file__).resolve().parent.parent.parent.parent
    
    # 准备数据集
    if prepare_dataset:
        output_dir = resolve_path(config['dataset']['output_dir'])
        ensure_dir(output_dir)
        
        logger.info("正在准备数据集...")
        config_path = resolve_path(config.get('_config_path', None))
        if config_path is None:
            from wot_ai.utils.paths import get_project_root
            project_root = get_project_root()
            config_path = project_root / "wot_ai" / "train_yolo" / "train_config.yaml"
        
        # 优先使用直接指定的路径
        source_images = None
        source_labels = None
        source_dir = None
        
        if 'source_images' in config['dataset'] and config['dataset']['source_images']:
            source_images = resolve_path(config['dataset']['source_images'])
            logger.info(f"使用指定的源图像目录: {source_images}")
        if 'source_labels' in config['dataset'] and config['dataset']['source_labels']:
            source_labels = resolve_path(config['dataset']['source_labels'])
            logger.info(f"使用指定的源标签目录: {source_labels}")
        
        if source_images is None or source_labels is None:
            if 'source_dir' in config['dataset']:
                source_dir = resolve_path(config['dataset']['source_dir'])
                logger.info(f"使用源数据目录: {source_dir}")
            else:
                logger.error("配置文件中必须指定 source_dir 或同时指定 source_images 和 source_labels")
                return False
        
        if not PrepareMinimapDataset(
            source_dir=source_dir,
            source_images=source_images,
            source_labels=source_labels,
            output_dir=output_dir,
            train_ratio=config['dataset'].get('train_ratio', 0.8),
            random_seed=config['dataset'].get('random_seed', None),
            validate_labels=True,
            config_path=config_path
        ):
            logger.error("数据集准备失败")
            return False
    
    # 检查数据集结构
    dataset_dir = resolve_path(config['dataset']['output_dir'])
    
    if not CheckDatasetStructure(dataset_dir):
        logger.error("数据集结构检查失败")
        return False
    
    # 生成或检查 dataset.yaml
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        logger.info("dataset.yaml 不存在，正在生成...")
        # 确保类别从配置文件加载
        if 'classes' not in config or not config['classes']:
            config_path = Path(config.get('_config_path', None)) if config.get('_config_path') else None
            config['classes'] = LoadClassesFromConfig(config_path)
        GenerateDatasetYaml(dataset_dir, config['classes'])
    else:
        logger.info(f"使用现有的 dataset.yaml: {yaml_path}")
    
    # 加载模型
    model_name = config['model']['name']
    logger.info(f"加载模型: {model_name}")
    model = YOLO(model_name)
    
    # 准备训练参数
    train_params = {
        'data': str(yaml_path),
        'imgsz': config['model']['imgsz'],
        'epochs': config['training']['epochs'],
        'batch': config['training']['batch'],
        'workers': config['training']['workers'],
        'device': config['model']['device'],
        'project': config['output']['project'],
        'name': config['output']['name'],
        'save': config['output']['save'],
        'save_period': config['output']['save_period'],
        'val': config['output']['val'],
        'verbose': True
    }
    
    # 优化器和学习率参数
    if 'optimizer' in config['training']:
        train_params['optimizer'] = config['training']['optimizer']
    if 'lr0' in config['training']:
        train_params['lr0'] = config['training']['lr0']
    if 'lrf' in config['training']:
        train_params['lrf'] = config['training']['lrf']
    if 'warmup_epochs' in config['training']:
        train_params['warmup_epochs'] = config['training']['warmup_epochs']
    if 'patience' in config['training']:
        train_params['patience'] = config['training']['patience']
    if 'dropout' in config['training']:
        train_params['dropout'] = config['training']['dropout']
    
    # 损失权重配置（如果存在）
    if 'loss' in config['training']:
        loss_config = config['training']['loss']
        if isinstance(loss_config, dict):
            # 损失权重直接作为 train_params 的参数传入（Ultralytics YOLO v8 使用 cls, box, dfl）
            if 'cls' in loss_config:
                train_params['cls'] = loss_config['cls']
            if 'box' in loss_config:
                train_params['box'] = loss_config['box']
            if 'dfl' in loss_config:
                train_params['dfl'] = loss_config['dfl']
            
            logger.info(f"损失权重配置: cls={loss_config.get('cls', 0.5)}, "
                       f"box={loss_config.get('box', 7.5)}, "
                       f"dfl={loss_config.get('dfl', 1.5)}")
    
    # 数据增强参数
    if config['training'].get('augment', True):
        train_params.update({
            'augment': True,
            'hsv_h': config['augmentation'].get('hsv_h', 0.015),
            'hsv_s': config['augmentation'].get('hsv_s', 0.7),
            'hsv_v': config['augmentation'].get('hsv_v', 0.4),
            'degrees': config['augmentation'].get('degrees', 5),
            'scale': config['augmentation'].get('scale', 0.5),
            'translate': config['augmentation'].get('translate', 0.1),
            'fliplr': config['augmentation'].get('fliplr', 0.2),
            'mosaic': config['training'].get('mosaic', 1.0),
            'copy_paste': config['training'].get('copy_paste', 0.1),
            'mixup': config['training'].get('mixup', 0.0),
            'close_mosaic': config['training'].get('close_mosaic', 0)
        })
    
    # 开始训练
    logger.info("开始训练 YOLO 模型...")
    logger.info(f"训练参数: epochs={train_params['epochs']}, batch={train_params['batch']}, imgsz={train_params['imgsz']}")
    if 'optimizer' in train_params:
        logger.info(f"优化器: {train_params['optimizer']}, lr0={train_params.get('lr0', 'default')}, "
                   f"lrf={train_params.get('lrf', 'default')}")
    if config['training'].get('augment', True):
        logger.info(f"数据增强: mosaic={train_params.get('mosaic', 1.0)}, "
                   f"copy_paste={train_params.get('copy_paste', 0.1)}, "
                   f"mixup={train_params.get('mixup', 0.0)}, "
                   f"close_mosaic={train_params.get('close_mosaic', 0)}")
    
    try:
        results = model.train(**train_params)
        
        logger.info("\n" + "=" * 60)
        logger.info("训练完成！结果摘要：")
        if hasattr(results, 'save_dir'):
            logger.info(f"训练日志保存目录: {results.save_dir}")
            best_model = Path(results.save_dir) / "weights" / "best.pt"
            if best_model.exists():
                logger.info(f"最佳模型路径: {best_model}")
        logger.info("=" * 60)
        
        return True
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练 YOLO 小地图检测模型")
    parser.add_argument(
        "--config",
        type=str,
        default="wot_ai/train_yolo/train_config.yaml",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--no_prepare",
        action="store_true",
        help="跳过数据集准备（使用已准备好的数据集）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖模型名称（如 yolo11n.pt）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆盖训练轮数"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="覆盖批次大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="覆盖设备（cpu, 0, 1等）"
    )
    
    args = parser.parse_args()
    
    # 加载配置（使用统一路径解析）
    config_path = resolve_path(args.config)
    
    config = LoadConfig(config_path)
    
    # 命令行参数覆盖
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch:
        config['training']['batch'] = args.batch
    if args.device:
        config['model']['device'] = args.device
    
    # 执行训练
    success = TrainYOLO(config, prepare_dataset=not args.no_prepare)
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

