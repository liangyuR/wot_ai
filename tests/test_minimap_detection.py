#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测功能测试
测试新的 MinimapDetector（YOLO/传统方案）
"""

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from pathlib import Path
from typing import Dict, Optional, Tuple

# 设置 Python 路径，确保可以正确导入项目模块
try:
    from wot_ai.utils.paths import setup_python_path
    setup_python_path()
except ImportError:
    # 如果 wot_ai 包不可用，手动设置路径
    import sys
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# 统一导入机制
from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple
setup_python_path()

# 导入新的检测器
YOLODetector, _ = try_import_multiple([
    'wot_ai.game_modules.navigation.core.minimap_detector',
    'game_modules.navigation.core.minimap_detector',
    'navigation.core.minimap_detector'
])
if YOLODetector is None:
    from navigation.core.minimap_detector import YOLODetector

TraditionalDetector, _ = try_import_multiple([
    'wot_ai.game_modules.navigation.core.minimap_detector',
    'game_modules.navigation.core.minimap_detector',
    'navigation.core.minimap_detector'
])
if TraditionalDetector is None:
    from navigation.core.minimap_detector import TraditionalDetector

MinimapDetector, _ = try_import_multiple([
    'wot_ai.game_modules.navigation.core.minimap_detector',
    'game_modules.navigation.core.minimap_detector',
    'navigation.core.minimap_detector'
])
if MinimapDetector is None:
    from navigation.core.minimap_detector import MinimapDetector


def FindModelPath(script_dir: Path) -> Optional[Path]:
    """查找 YOLO 模型路径"""
    from wot_ai.utils.paths import get_models_dir
    models_dir = get_models_dir()
    possible_paths = [
        models_dir / "yolo" / "minimap" / "best.pt",
        models_dir / "yolo" / "minimap" / "minimap_yolo.pt",
        script_dir.parent / "navigation" / "train" / "model" / "minimap_yolo.pt",
        script_dir.parent.parent / "train" / "model" / "minimap_yolo.pt",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def VisualizeResults(minimap: np.ndarray, results: Dict, detector_name: str) -> np.ndarray:
    """可视化检测结果"""
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


def TestTraditionalDetector(minimap: np.ndarray) -> Tuple[bool, Dict, np.ndarray]:
    """测试传统检测器"""
    print("\n" + "=" * 60)
    print("测试传统检测器 (TraditionalDetector)")
    print("=" * 60)
    
    try:
        detector = TraditionalDetector()
        print("✓ 传统检测器创建成功")
        
        results = detector.Detect(minimap)
        print("✓ 检测完成")
        
        # 显示结果
        print(f"  己方位置: {results.get('self_pos')}")
        print(f"  基地位置: {results.get('flag_pos')}")
        print(f"  障碍物数量: {len(results.get('obstacles', []))}")
        print(f"  道路数量: {len(results.get('roads', []))}")
        
        display = VisualizeResults(minimap, results, "Traditional Detector")
        
        success = results.get('self_pos') is not None or results.get('flag_pos') is not None
        return success, results, display
        
    except Exception as e:
        print(f"✗ 传统检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, minimap.copy()


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
    print("=" * 60)
    print("开始测试小地图检测功能（新版本）")
    print("=" * 60)
    
    # 1. 加载测试图片
    script_dir = Path(__file__).resolve().parent
    minimap_path = script_dir.parent / "path_planning" / "maps" / "01.png"
    
    if not minimap_path.exists():
        print(f"错误：测试图片不存在: {minimap_path}")
        return False
    
    minimap = cv2.imread(str(minimap_path))
    if minimap is None:
        print(f"错误：无法读取测试图片: {minimap_path}")
        return False
    
    h, w = minimap.shape[:2]
    print(f"\n✓ 测试图片加载成功: {w}x{h}")
    
    # 2. 配置小地图区域
    minimap_region = {
        'x': 0,
        'y': 0,
        'width': w,
        'height': h
    }
    
    # 3. 查找模型路径
    model_path = FindModelPath(script_dir)
    if model_path:
        print(f"\n✓ 找到模型文件: {model_path}")
    else:
        print("\n⚠ 未找到模型文件，将跳过 YOLO 相关测试")
    
    # 4. 测试传统检测器
    trad_success, trad_results, trad_display = TestTraditionalDetector(minimap)
    
    # 5. 测试 YOLO 检测器（如果模型存在）
    yolo_success = False
    yolo_results = {}
    yolo_display = None
    if model_path:
        yolo_success, yolo_results, yolo_display = TestYOLODetector(minimap, model_path)
    
    # 6. 测试兼容层检测器（如果模型存在）
    compat_success = False
    compat_results = {}
    compat_display = None
    if model_path:
        compat_success, compat_results, compat_display = TestMinimapDetector(
            minimap, model_path, minimap_region
        )
    
    # 7. 保存和显示结果
    print("\n" + "=" * 60)
    print("保存可视化结果")
    print("=" * 60)
    
    output_dir = script_dir
    try:
        # 保存传统检测器结果
        trad_output = output_dir / "test_traditional_result.jpg"
        cv2.imwrite(str(trad_output), trad_display)
        print(f"✓ 传统检测器结果已保存: {trad_output}")
        
        # 保存 YOLO 检测器结果（如果有）
        if yolo_display is not None:
            yolo_output = output_dir / "test_yolo_result.jpg"
            cv2.imwrite(str(yolo_output), yolo_display)
            print(f"✓ YOLO检测器结果已保存: {yolo_output}")
        
        # 保存兼容层检测器结果（如果有）
        if compat_display is not None:
            compat_output = output_dir / "test_compat_result.jpg"
            cv2.imwrite(str(compat_output), compat_display)
            print(f"✓ 兼容层检测器结果已保存: {compat_output}")
        
        # 显示结果（显示传统检测器的结果）
        cv2.imshow("Traditional Detector Result", trad_display)
        if yolo_display is not None:
            cv2.imshow("YOLO Detector Result", yolo_display)
        if compat_display is not None:
            cv2.imshow("Compatible Detector Result", compat_display)
        
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"⚠ 保存/显示结果出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. 测试结果评估
    print("\n" + "=" * 60)
    print("测试结果评估")
    print("=" * 60)
    
    all_success = True
    
    if trad_success:
        print("✓ 传统检测器测试: 通过")
    else:
        print("✗ 传统检测器测试: 失败（未检测到关键元素）")
        all_success = False
    
    if model_path:
        if yolo_success:
            print("✓ YOLO检测器测试: 通过")
        else:
            print("✗ YOLO检测器测试: 失败（未检测到关键元素）")
            all_success = False
        
        if compat_success:
            print("✓ 兼容层检测器测试: 通过")
        else:
            print("✗ 兼容层检测器测试: 失败（未检测到关键元素）")
            all_success = False
    else:
        print("⚠ YOLO相关测试: 跳过（未找到模型文件）")
    
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
