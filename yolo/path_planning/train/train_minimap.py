#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测模型训练脚本
"""

from pathlib import Path
import argparse
from ultralytics import YOLO
try:
    from path_planning.utils.logger import SetupLogger
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.logger import SetupLogger

logger = SetupLogger(__name__)


def TrainMinimapModel(data_yaml: str, epochs: int = 100, imgsz: int = 640, 
                     model: str = "yolo11n.pt", batch: int = 16, device: str = "cpu") -> bool:
    """
    训练小地图检测模型
    
    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        imgsz: 图像尺寸
        model: 基础模型（yolo11n.pt, yolo11s.pt, yolo11m.pt等）
        batch: 批次大小
        device: 设备（cpu, 0, 1等）
    
    Returns:
        是否成功
    """
    data_path = Path(data_yaml)
    if not data_path.exists():
        logger.error(f"数据集配置文件不存在: {data_yaml}")
        return False
    
    try:
        logger.info(f"开始训练小地图检测模型")
        logger.info(f"数据集配置: {data_yaml}")
        logger.info(f"训练轮数: {epochs}, 图像尺寸: {imgsz}, 批次大小: {batch}")
        
        # 加载模型
        yolo_model = YOLO(model)
        
        # 训练
        results = yolo_model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project="yolo/path_planning/train",
            name="minimap_detection",
            save=True,
            save_period=10,
            val=True
        )
        
        logger.info("训练完成")
        logger.info(f"最佳模型保存在: {yolo_model.trainer.best}")
        
        return True
    except Exception as e:
        logger.error(f"训练失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练小地图检测模型")
    parser.add_argument("--data", type=str, required=True, help="数据集配置文件路径（data.yaml）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="基础模型")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=str, default="cpu", help="设备（cpu, 0, 1等）")
    
    args = parser.parse_args()
    
    TrainMinimapModel(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model=args.model,
        batch=args.batch,
        device=args.device
    )


if __name__ == "__main__":
    main()

