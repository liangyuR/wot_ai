#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测引擎：封装 YOLO 模型检测逻辑
"""

from pathlib import Path
from typing import List, Optional, Dict
import logging
import numpy as np

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DetectionEngine:
    """检测引擎：封装 YOLO 检测逻辑"""
    
    def __init__(self, model_path: str, base_dir: Optional[Path] = None):
        """
        初始化检测引擎
        
        Args:
            model_path: 模型文件路径（相对或绝对）
            base_dir: 基础目录（用于解析相对路径）
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
        
        model_path_obj = Path(model_path)
        if model_path_obj.is_absolute():
            self.model_path_ = str(model_path_obj)
        else:
            self.model_path_ = str(base_dir / model_path_obj)
        
        self.model_ = None
        self.base_dir_ = base_dir
    
    def LoadModel(self) -> bool:
        """
        加载 YOLO 模型
        
        Returns:
            是否加载成功
        """
        try:
            model_path_obj = Path(self.model_path_)
            if not model_path_obj.exists():
                logger.error(f"模型文件不存在: {self.model_path_}")
                return False
            
            logger.info(f"加载模型: {model_path_obj.absolute()}")
            self.model_ = YOLO(str(model_path_obj))
            logger.info("模型加载成功")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def Detect(self, frame: np.ndarray, confidence_threshold: float = 0.25) -> List[Dict]:
        """
        检测帧中的目标
        
        Args:
            frame: 输入帧（BGR格式，numpy数组）
            confidence_threshold: 置信度阈值
        
        Returns:
            检测结果列表，每个结果包含：
            - class: 类别ID
            - confidence: 置信度
            - bbox: [x1, y1, x2, y2]
        """
        if self.model_ is None:
            logger.error("模型未加载，请先调用 LoadModel()")
            return []
        
        try:
            # YOLO 接受 numpy 数组或 PIL Image
            results = self.model_(frame, conf=confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detections.append({
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    })
            
            return detections
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []
    
    def GetBestTarget(self, detections: List[Dict], target_class: Optional[int] = None) -> Optional[tuple[int, int]]:
        """
        从检测结果中选择最佳目标
        
        Args:
            detections: 检测结果列表
            target_class: 目标类别ID（如果None则选择置信度最高的）
        
        Returns:
            目标坐标 (x, y)，如果未找到则返回None
        """
        if not detections:
            return None
        
        # 如果指定了类别，优先选择该类别的目标
        if target_class is not None:
            class_detections = [d for d in detections if d['class'] == target_class]
            if class_detections:
                detections = class_detections
        
        # 选择置信度最高的目标
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        # 计算bbox中心
        bbox = best_detection['bbox']
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        return center_x, center_y

