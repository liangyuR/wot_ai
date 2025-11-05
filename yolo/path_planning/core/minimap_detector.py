#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图识别模块：从屏幕截图中检测小地图元素
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
import logging

from ultralytics import YOLO
try:
    from path_planning.core.position_smoother import PositionSmoother
    from path_planning.utils.logger import SetupLogger
except ImportError:
    # 相对导入
    from .position_smoother import PositionSmoother
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.logger import SetupLogger

logger = SetupLogger(__name__)


# 检测类别映射
CLASS_MAPPING = {
    0: 'minimap_self',
    1: 'minimap_ally',
    2: 'minimap_enemy',
    3: 'flag_base',
    4: 'minimap_obstacle',
    5: 'minimap_road'
}


class MinimapDetector:
    """小地图检测器"""
    
    def __init__(self, model_path: str, minimap_region: Dict, base_dir: Optional[Path] = None):
        """
        初始化小地图检测器
        
        Args:
            model_path: YOLO模型路径
            minimap_region: 小地图区域配置 {'x': int, 'y': int, 'width': int, 'height': int}
            base_dir: 基础目录（用于解析相对路径）
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
        
        model_path_obj = Path(model_path)
        if model_path_obj.is_absolute():
            self.model_path_ = str(model_path_obj)
        else:
            self.model_path_ = str(base_dir / model_path_obj)
        
        self.minimap_region_ = minimap_region
        self.model_ = None
        self.base_dir_ = base_dir
        
        # 位置平滑器
        self.self_pos_smoother_ = PositionSmoother(window=3)
        self.flag_pos_smoother_ = PositionSmoother(window=3)
    
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
            
            logger.info(f"加载小地图检测模型: {model_path_obj.absolute()}")
            self.model_ = YOLO(str(model_path_obj))
            logger.info("小地图检测模型加载成功")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def ExtractMinimap(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从完整屏幕帧中提取小地图区域
        
        Args:
            frame: 完整屏幕帧（BGR格式）
        
        Returns:
            小地图区域（BGR格式），如果区域无效则返回 None
        """
        try:
            x = self.minimap_region_['x']
            y = self.minimap_region_['y']
            width = self.minimap_region_['width']
            height = self.minimap_region_['height']
            
            # 检查边界
            if x < 0 or y < 0:
                logger.warning(f"小地图区域坐标无效: ({x}, {y})")
                return None
            
            if x + width > frame.shape[1] or y + height > frame.shape[0]:
                logger.warning(f"小地图区域超出屏幕范围")
                return None
            
            minimap = frame[y:y+height, x:x+width]
            return minimap
        except Exception as e:
            logger.error(f"提取小地图失败: {e}")
            return None
    
    def Detect(self, frame: np.ndarray, confidence_threshold: float = 0.25) -> Dict:
        """
        检测小地图中的元素
        
        Args:
            frame: 完整屏幕帧（BGR格式）
            confidence_threshold: 置信度阈值
        
        Returns:
            检测结果字典：
            {
                'self_pos': (x, y) or None,
                'flag_pos': (x, y) or None,
                'obstacles': [(x1, y1, x2, y2), ...],
                'roads': [(x1, y1, x2, y2), ...]
            }
        """
        if self.model_ is None:
            logger.error("模型未加载，请先调用 LoadModel()")
            return self._EmptyResult()
        
        # 提取小地图区域
        minimap = self.ExtractMinimap(frame)
        if minimap is None:
            return self._EmptyResult()
        
        try:
            # YOLO检测
            results = self.model_(minimap, conf=confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detections.append({
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2] 相对于小地图区域
                    })
            
            # 解析检测结果
            return self.ParseDetections(detections)
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return self._EmptyResult()
    
    def ParseDetections(self, detections: List[Dict]) -> Dict:
        """
        解析检测结果，转换为结构化格式
        
        Args:
            detections: YOLO检测结果列表
        
        Returns:
            结构化检测结果
        """
        result = {
            'self_pos': None,
            'flag_pos': None,
            'obstacles': [],
            'roads': []
        }
        
        for det in detections:
            class_id = det['class']
            bbox = det['bbox']
            
            # 计算bbox中心坐标（相对于小地图）
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            class_name = CLASS_MAPPING.get(class_id, 'unknown')
            
            if class_name == 'minimap_self':
                # 平滑处理
                raw_pos = (center_x, center_y)
                result['self_pos'] = self.self_pos_smoother_.Smooth(raw_pos)
            elif class_name == 'flag_base':
                raw_pos = (center_x, center_y)
                result['flag_pos'] = self.flag_pos_smoother_.Smooth(raw_pos)
            elif class_name == 'minimap_obstacle':
                result['obstacles'].append((bbox[0], bbox[1], bbox[2], bbox[3]))
            elif class_name == 'minimap_road':
                result['roads'].append((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        return result
    
    def _EmptyResult(self) -> Dict:
        """返回空的检测结果"""
        return {
            'self_pos': None,
            'flag_pos': None,
            'obstacles': [],
            'roads': []
        }
    
    def ResetSmoothers(self):
        """重置位置平滑器"""
        self.self_pos_smoother_.Reset()
        self.flag_pos_smoother_.Reset()

