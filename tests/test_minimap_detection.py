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

try:
    from wot_ai.utils.paths import setup_python_path
    setup_python_path()
except ImportError:
    import sys
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple
setup_python_path()

from wot_ai.game_modules.vision.detection.minimap_detector import MinimapDetector
from wot_ai.game_modules.vision.detection.minimap_anchor_detector import MinimapAnchorDetector


def FindModelPath(script_dir: Path) -> Optional[Path]:
    """查找 YOLO 模型路径（固定在 ../wot_ai/model/）"""
    model_path = script_dir.parent / "wot_ai" / "model" / "minimap_yolo.pt"
    if model_path.exists():
        return model_path
    return None

"""可视化检测结果"""
def VisualizeResults(minimap: np.ndarray, results: Dict, detector_name: str) -> np.ndarray:
    display = minimap.copy()
    
    # 绘制己方位置
    if results.get('self_pos'):
        cx, cy = results['self_pos']
        cx, cy = int(cx), int(cy)
        cv2.circle(display, (cx, cy), 8, (0, 255, 255), -1)  # 黄色实心圆
        cv2.circle(display, (cx, cy), 12, (0, 255, 255), 2)  # 黄色外圈
        cv2.putText(display, "Self", (cx + 15, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 绘制基地位置
    if results.get('flag_pos'):
        cx, cy = results['flag_pos']
        cx, cy = int(cx), int(cy)
        cv2.rectangle(display, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 255), 2)
        cv2.putText(display, "Flag", (cx + 15, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 绘制障碍物
    for obstacle in results.get('obstacles', []):
        x1, y1, x2, y2 = map(int, obstacle)
        cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)
    
    # 绘制道路
    for road in results.get('roads', []):
        x1, y1, x2, y2 = map(int, road)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # 添加检测器名称
    cv2.putText(display, detector_name, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display

def TestYOLODetector(minimap: np.ndarray, model_path: Path) -> Tuple[bool, Dict, np.ndarray]:
    """测试 YOLO 检测器"""
    print("\n" + "=" * 60)
    print("测试 YOLO 检测器 (YOLODetector)")
    print("=" * 60)
    
    try:
        detector = YOLODetector(str(model_path), confidence_threshold=0.25)
        print(f"✓ YOLO检测器创建成功")
        
        if not detector.LoadModel():
            print("✗ 模型加载失败")
            return False, {}, minimap.copy()
        
        print("✓ 模型加载成功")
        
        results = detector.Detect(minimap)
        print("✓ 检测完成")
        
        # 显示结果
        print(f"  己方位置: {results.get('self_pos')}")
        print(f"  基地位置: {results.get('flag_pos')}")
        print(f"  障碍物数量: {len(results.get('obstacles', []))}")
        print(f"  道路数量: {len(results.get('roads', []))}")
        
        display = VisualizeResults(minimap, results, "YOLO Detector")
        
        success = results.get('self_pos') is not None or results.get('flag_pos') is not None
        return success, results, display
        
    except Exception as e:
        print(f"✗ YOLO检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, minimap.copy()


def TestMinimapDetector(minimap: np.ndarray, model_path: Optional[Path], 
                       minimap_region: Dict) -> Tuple[bool, Dict, np.ndarray]:
    """测试兼容层检测器"""
    print("\n" + "=" * 60)
    print("测试兼容层检测器 (MinimapDetector)")
    print("=" * 60)
    
    if model_path is None:
        print("⚠ 未找到模型文件，跳过 YOLO 兼容层测试")
        return False, {}, minimap.copy()
    
    try:
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir.parent
        
        detector = MinimapDetector(
            model_path=str(model_path),
            minimap_region=minimap_region,
            base_dir=base_dir
        )
        print("✓ 兼容层检测器创建成功")
        
        if not detector.LoadModel():
            print("✗ 模型加载失败")
            return False, {}, minimap.copy()
        
        print("✓ 模型加载成功")
        
        # 创建完整屏幕帧（将小地图放在指定位置）
        h, w = minimap.shape[:2]
        full_frame = np.zeros((minimap_region['y'] + h, minimap_region['x'] + w, 3), dtype=np.uint8)
        full_frame[minimap_region['y']:minimap_region['y']+h, 
                  minimap_region['x']:minimap_region['x']+w] = minimap
        
        results = detector.Detect(full_frame, confidence_threshold=0.25)
        print("✓ 检测完成")
        
        # 显示结果
        print(f"  己方位置: {results.get('self_pos')}")
        print(f"  基地位置: {results.get('flag_pos')}")
        print(f"  障碍物数量: {len(results.get('obstacles', []))}")
        print(f"  道路数量: {len(results.get('roads', []))}")
        
        display = VisualizeResults(minimap, results, "MinimapDetector (Compatible)")
        
        success = results.get('self_pos') is not None or results.get('flag_pos') is not None
        return success, results, display
        
    except Exception as e:
        print(f"✗ 兼容层检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, minimap.copy()


def TestMinimapDetection():
    """测试小地图检测功能"""
    logger.info("=" * 60)
    logger.info("开始测试小地图检测功能（YOLO）")
    logger.info("=" * 60)

    # 1. 加载测试图片
    script_dir = Path(__file__).resolve().parent
    minimap_path = script_dir / "minimap.png"
    if not minimap_path.exists():
        logger.error(f"错误：测试图片不存在: {minimap_path}")
        return False
    
    minimap = cv2.imread(str(minimap_path))
    if minimap is None:
        logger.error(f"错误：无法读取测试图片: {minimap_path}")
        return False
    
    h, w = minimap.shape[:2]
    logger.info(f"测试图片加载成功: {w}x{h}")
    
    # 2. 配置小地图区域
    minimap_template_path = script_dir / "minimap_border.png"
    if not minimap_template_path.exists():
        logger.error(f"错误：小地图模板不存在: {minimap_template_path}")
        return False

    anchor_detector = MinimapAnchorDetector(template_path=str(minimap_template_path), debug=True)
    anchor_pos = anchor_detector.detect(minimap)
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
    logger.info("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True

    # 3. 查找模型路径. wot_ai/models/minimap_dectect.pt
    model_path = Path("wot_ai/models/minimap_dectect_yolo11m.pt")
    logger.info(f"模型路径: {model_path}")
    if not model_path.exists():
        logger.error(f"错误：模型文件不存在: {model_path}")
        return False
    
    # 5. 保存和显示结果
    logger.info("保存可视化结果")
    
    output_dir = script_dir
    try:
        # 保存 YOLO 检测器结果
        yolo_output = output_dir / "test_yolo_result.jpg"
        cv2.imwrite(str(yolo_output), yolo_display)
        logger.info(f"✓ YOLO检测器结果已保存: {yolo_output}")
        
        # 显示结果
        cv2.imshow("YOLO Detector Result", yolo_display)
        logger.info("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"⚠ 保存/显示结果出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. 测试结果评估
    logger.info("=" * 60)
    logger.info("测试结果评估")
    logger.info("=" * 60)
    
    all_success = True

    if all_success:
        print("\n✓✓✓ 所有测试通过！")
    else:
        print("\n⚠ 部分测试未通过（可能是图片中确实没有目标元素）")
    
    print("=" * 60)
    
    return all_success


def main():
    """主函数"""
    success = TestMinimapDetection()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
