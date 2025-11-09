#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图路径规划演示脚本
支持手动指定起终点或使用YOLO模型自动检测
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from wot_ai.utils.paths import setup_python_path
setup_python_path()

from wot_ai.game_modules.navigation.runtime.navigation_loop import NavigationRuntime


def ParseArgs():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="小地图路径规划演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 手动指定起终点
  python demo.py --map_id fishermans_bay --maps_dir data/maps \\
    --minimap_img examples/minimap.png --grid 64 64 --inflate_px 4 \\
    --start 300 260 --goal 480 480 --out outputs/demo_result.png

  # 使用YOLO模型自动检测起终点
  python demo.py --map_id fishermans_bay --maps_dir data/maps \\
    --minimap_img examples/minimap.png --grid 64 64 --inflate_px 4 \\
    --model_path data/models/training/minimap/xxx/best.pt \\
    --out outputs/demo_result.png
        """
    )
    
    parser.add_argument("--map_id", type=str, required=True,
                       help="地图ID（对应data/maps下的子目录名）")
    parser.add_argument("--maps_dir", type=str, default="data/maps",
                       help="地图数据目录（默认: data/maps）")
    parser.add_argument("--minimap_img", type=str, required=True,
                       help="小地图图像路径（或整屏截图）")
    parser.add_argument("--region", type=str, default=None,
                       help="小地图区域（格式: x,y,width,height），如果minimap_img已是裁剪后的小地图则不需要")
    parser.add_argument("--grid", type=int, nargs=2, default=[64, 64],
                       metavar=("W", "H"), help="栅格尺寸（默认: 64 64）")
    parser.add_argument("--inflate_px", type=int, default=4,
                       help="膨胀像素数（默认: 4）")
    parser.add_argument("--start", type=int, nargs=2, default=None,
                       metavar=("X", "Y"), help="起点坐标（小地图像素坐标）")
    parser.add_argument("--goal", type=int, nargs=2, default=None,
                       metavar=("X", "Y"), help="终点坐标（小地图像素坐标）")
    parser.add_argument("--model_path", type=str, default=None,
                       help="YOLO模型路径（用于自动检测起终点）")
    parser.add_argument("--out", type=str, default="outputs/demo_result.png",
                       help="输出图像路径（默认: outputs/demo_result.png）")
    parser.add_argument("--performance", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="性能模式（默认: medium）")
    
    return parser.parse_args()


def LoadImage(img_path: str, region: str = None) -> np.ndarray:
    """加载图像，如果指定了region则裁剪"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    if region:
        x, y, w, h = map(int, region.split(','))
        img = img[y:y+h, x:x+w]
    
    return img


def DrawResult(minimap: np.ndarray, grid: np.ndarray, path: list,
               start: tuple = None, goal: tuple = None,
               grid_size: tuple = None) -> np.ndarray:
    """在图像上绘制栅格地图、路径和起终点"""
    result = minimap.copy()
    H, W = minimap.shape[:2]
    
    if grid_size:
        scale_x = W / grid_size[0]
        scale_y = H / grid_size[1]
    else:
        scale_x = scale_y = 1.0
    
    # 绘制障碍物（红色半透明）
    if grid is not None:
        grid_vis = cv2.resize(grid.astype(np.uint8) * 255, (W, H), 
                              interpolation=cv2.INTER_NEAREST)
        mask = grid_vis > 0
        result[mask] = cv2.addWeighted(result[mask], 0.5, 
                                      np.array([0, 0, 255], dtype=np.uint8), 0.5, 0)[mask]
    
    # 绘制路径（绿色线条）
    if path and len(path) > 1:
        path_pixels = []
        for px, py in path:
            px_pixel = int(px * scale_x)
            py_pixel = int(py * scale_y)
            path_pixels.append((px_pixel, py_pixel))
        
        for i in range(len(path_pixels) - 1):
            cv2.line(result, path_pixels[i], path_pixels[i+1], 
                    (0, 255, 0), 2)
    
    # 绘制起点（蓝色圆点）
    if start:
        sx, sy = start
        if grid_size:
            sx = int(sx * scale_x)
            sy = int(sy * scale_y)
        cv2.circle(result, (sx, sy), 5, (255, 0, 0), -1)
    
    # 绘制终点（红色圆圈）
    if goal:
        gx, gy = goal
        if grid_size:
            gx = int(gx * scale_x)
            gy = int(gy * scale_y)
        cv2.circle(result, (gx, gy), 8, (0, 0, 255), 2)
    
    return result


def Main():
    """主函数"""
    args = ParseArgs()
    
    # 创建输出目录
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载图像
    print(f"加载图像: {args.minimap_img}")
    frame = LoadImage(args.minimap_img, args.region)
    
    # 确定小地图区域
    if args.region:
        x, y, w, h = map(int, args.region.split(','))
        minimap_region = {"x": x, "y": y, "width": w, "height": h}
    else:
        H, W = frame.shape[:2]
        minimap_region = {"x": 0, "y": 0, "width": W, "height": H}
    
    # 确定模型路径
    model_path = args.model_path
    if model_path is None:
        # 如果没有提供模型，必须手动指定起终点
        if args.start is None or args.goal is None:
            print("错误: 未提供模型路径时，必须手动指定 --start 和 --goal")
            return 1
        # 使用一个虚拟模型路径（实际不会加载）
        model_path = "dummy.pt"
    
    # 初始化导航运行时
    print(f"初始化导航运行时: map_id={args.map_id}, grid={args.grid}")
    try:
        runtime = NavigationRuntime(
            map_id=args.map_id,
            minimap_region=minimap_region,
            model_path=model_path,
            grid_size=tuple(args.grid),
            maps_dir=args.maps_dir,
            inflate_px=args.inflate_px,
            performance=args.performance,
            base_dir=Path(__file__).parent
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return 1
    
    # 处理帧
    print("处理帧...")
    grid, path = runtime.step(frame)
    
    if grid is None:
        print("错误: 处理失败")
        return 1
    
    # 确定起终点
    start = None
    goal = None
    
    if args.start and args.goal:
        # 手动指定
        start = args.start
        goal = args.goal
    else:
        # 从检测结果获取（需要重新检测）
        detections = runtime.detector_.Detect(frame)
        if detections.get('self_pos'):
            start = detections['self_pos']
        if detections.get('flag_pos'):
            goal = detections['flag_pos']
    
    if start is None or goal is None:
        print("警告: 无法确定起终点，路径可能为空")
    
    # 绘制结果
    print("绘制结果...")
    result_img = DrawResult(
        frame, grid, path, 
        start=start, goal=goal,
        grid_size=tuple(args.grid)
    )
    
    # 保存结果
    cv2.imwrite(str(out_path), result_img)
    print(f"结果已保存: {out_path}")
    print(f"路径长度: {len(path)} 个点")
    
    return 0


if __name__ == "__main__":
    sys.exit(Main())

