#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测功能测试
测试 CvMinimapDetector 的地图检测功能
"""

import cv2
import numpy as np
from pathlib import Path

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

# 导入检测器
from yolo.core.cv_minimap_detector import CvMinimapDetector


def TestMinimapDetection():
    """测试小地图检测功能"""
    print("=" * 60)
    print("开始测试小地图检测功能")
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
    
    # 2. 配置检测器
    minimap_region = {
        'x': 0,
        'y': 0,
        'width': w,
        'height': h
    }
    
    detect_config = {
        'detect_self': True,        # 检测己方位置
        'detect_allies': False,     # 检测队友
        'detect_enemies': False,    # 检测敌人
        'detect_ally_base': False,  # 检测我方基地
        'detect_enemy_base': True   # 检测敌方基地
    }
    
    print("\n✓ 检测器配置完成")
    print(f"  检测己方位置: {detect_config['detect_self']}")
    print(f"  检测敌方基地: {detect_config['detect_enemy_base']}")
    
    # 3. 创建检测器
    try:
        detector = CvMinimapDetector(minimap_region, detect_config=detect_config)
        print("\n✓ 检测器创建成功")
    except Exception as e:
        print(f"\n✗ 检测器创建失败: {e}")
        return False
    
    # 4. 执行检测
    print("\n开始执行检测...")
    try:
        # 直接在小地图上检测（不通过 Parse，避免再次裁切）
        self_result = None
        if detect_config['detect_self']:
            self_result = detector.DetectSelf(minimap)
        
        bases = detector.DetectBases(minimap)
        
        # 组合结果
        if self_result:
            self_pos = (self_result[0], self_result[1])
            self_angle = self_result[2]
        else:
            self_pos = None
            self_angle = None
        
        results = {
            'self': {
                'pos': self_pos,
                'angle': self_angle
            },
            'allies': [],
            'enemies': [],
            'bases': bases
        }
        
        print("✓ 检测完成")
        
    except Exception as e:
        print(f"\n✗ 检测过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 验证检测结果
    print("\n" + "=" * 60)
    print("检测结果:")
    print("=" * 60)
    
    # 己方位置
    if results['self']['pos']:
        pos = results['self']['pos']
        angle = results['self']['angle']
        print(f"✓ 己方位置: ({pos[0]}, {pos[1]})")
        if angle is not None:
            print(f"✓ 己方朝向: {angle:.1f}°")
        else:
            print("  ⚠ 己方朝向: 未检测到")
    else:
        print("✗ 己方位置: 未检测到")
    
    # 敌方基地
    if results['bases']['enemy']:
        pos = results['bases']['enemy']
        print(f"✓ 敌方基地: ({pos[0]}, {pos[1]})")
    else:
        print("✗ 敌方基地: 未检测到")
    
    # 己方基地
    if results['bases']['ally']:
        pos = results['bases']['ally']
        print(f"✓ 己方基地: ({pos[0]}, {pos[1]})")
    else:
        print("  - 己方基地: 未检测（配置中已关闭）")
    
    print("=" * 60)
    
    # 6. 可视化检测结果
    print("\n生成可视化结果...")
    try:
        display = minimap.copy()
        
        # 绘制己方位置
        if results['self']['pos']:
            cx, cy = results['self']['pos']
            angle = results['self']['angle']
            
            cv2.circle(display, (cx, cy), 8, (0, 255, 255), -1)  # 黄色实心圆
            cv2.circle(display, (cx, cy), 12, (0, 255, 255), 2)  # 黄色外圈
            
            # 绘制方向线
            if angle is not None:
                line_length = 30
                dx = np.cos(np.radians(angle)) * line_length
                dy = np.sin(np.radians(angle)) * line_length
                ex = int(cx + dx)
                ey = int(cy + dy)
                cv2.line(display, (cx, cy), (ex, ey), (0, 255, 255), 2)
            
            cv2.putText(display, "Self", (cx + 15, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制敌方基地
        if results['bases']['enemy']:
            cx, cy = results['bases']['enemy']
            cv2.rectangle(display, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 255), 2)
            cv2.putText(display, "Enemy Base", (cx + 15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制己方基地
        if results['bases']['ally']:
            cx, cy = results['bases']['ally']
            cv2.rectangle(display, (cx - 10, cy - 10), (cx + 10, cy + 10), (255, 255, 255), 2)
            cv2.putText(display, "Ally Base", (cx + 15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存结果
        output_path = script_dir / "test_detection_result.jpg"
        cv2.imwrite(str(output_path), display)
        print(f"✓ 可视化结果已保存: {output_path}")
        
        # 显示结果
        cv2.imshow("Minimap Detection Test", display)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"⚠ 可视化过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 测试结果评估
    print("\n" + "=" * 60)
    print("测试结果评估:")
    print("=" * 60)
    
    success = True
    
    if not results['self']['pos']:
        print("✗ 测试失败: 未检测到己方位置")
        success = False
    else:
        print("✓ 己方位置检测: 通过")
    
    if not results['bases']['enemy']:
        print("✗ 测试失败: 未检测到敌方基地")
        success = False
    else:
        print("✓ 敌方基地检测: 通过")
    
    if success:
        print("\n✓✓✓ 所有测试通过！")
    else:
        print("\n✗✗✗ 部分测试失败")
    
    print("=" * 60)
    
    return success


def main():
    """主函数"""
    success = TestMinimapDetection()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

