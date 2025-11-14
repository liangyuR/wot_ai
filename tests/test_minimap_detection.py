#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测功能测试
测试新的 MinimapDetector（YOLO）
"""

import cv2  
import numpy as np 
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger
from wot_ai.game_modules.vision.detection.minimap_detector import MinimapDetector, MinimapDetectionResult
from wot_ai.game_modules.vision.detection.minimap_anchor_detector import MinimapAnchorDetector


def TestYOLODetector(minimap: np.ndarray, model_path: str, confidence_threshold: float = 0.25) -> Tuple[bool, MinimapDetectionResult, np.ndarray]:
    """测试 YOLO 检测器"""
    try:
        detector = MinimapDetector(model_path, conf_threshold=0.25)
        if not detector.LoadModel():
            logger.error("✗ 模型加载失败")
            return False, MinimapDetectionResult(None, None, []), minimap.copy()
        logger.info("✓ 模型加载成功")
        
        results = detector.Detect(minimap, debug=True)
        logger.info(f"检测结果: {results}")
    except Exception as e:
        print(f"✗ YOLO检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, MinimapDetectionResult(None, None, []), minimap.copy()

def TestMinimapDetection(image_path: str, template_path: str) -> np.ndarray:
    """测试小地图检测功能"""
    # 1. 加载测试图片
    origin_frame = cv2.imread(str(image_path))
    if origin_frame is None:
        logger.error(f"错误：无法读取测试图片: {image_path}")
        return False
    
    h, w = origin_frame.shape[:2]
    logger.info(f"测试图片加载成功: {w}x{h}")

    anchor_detector = MinimapAnchorDetector(template_path=str(template_path), debug=True)
    anchor_pos = anchor_detector.detect(origin_frame)
    if anchor_pos is None:
        logger.error("错误：无法检测到小地图锚点")
        return False
    x, y = anchor_pos
    minimap_region = {
        'x': x,
        'y': y,
        'width': w,
        'height': h
    }
    logger.info(f"小地图锚点: {x}, {y}, 小地图区域: {minimap_region}")
    minimap_frame = origin_frame[minimap_region['y']:minimap_region['y']+minimap_region['height'], minimap_region['x']:minimap_region['x']+minimap_region['width']]
    cv2.imshow("Minimap Frame", minimap_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return minimap_frame

def main():
    """主函数"""
    script_dir = Path(__file__).resolve().parent
    minimap_path = script_dir / "minimap.png"
    if not minimap_path.exists():
        logger.error(f"错误：测试图片不存在: {minimap_path}")
        return False

    minimap_template_path = script_dir / "minimap_border.png"
    if not minimap_template_path.exists():
        logger.error(f"错误：小地图模板不存在: {minimap_template_path}")
        return False

    model_path = script_dir / "best.pt"
    if not model_path.exists():
        logger.error(f"错误：模型不存在: {model_path}")
        return False

    minimap_frame = TestMinimapDetection(minimap_path, minimap_template_path)
    TestYOLODetector(minimap_frame, model_path, confidence_threshold=0.75)

if __name__ == "__main__":
    exit(main())
