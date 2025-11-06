#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KMeans二值化器测试脚本
用于测试和可视化二值化效果
"""

import cv2
import numpy as np
from pathlib import Path

try:
    from path_planning.core.minimap_binarizer import MinimapBinarizer
except ImportError:
    from core.minimap_binarizer import MinimapBinarizer


def main():
    """测试主函数"""
    # 1. 加载地图
    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "maps" / "01.png"
    
    if not map_path.exists():
        print(f"地图文件不存在: {map_path}")
        return
    
    image = cv2.imread(str(map_path))
    if image is None:
        print(f"无法读取地图文件: {map_path}")
        return
    
    print(f"地图加载成功: {image.shape[1]}x{image.shape[0]}")
    
    # 2. 测试不同聚类数量
    for n_clusters in [3, 5, 7]:
        print(f"\n测试聚类数量: {n_clusters}")
        
        # 创建二值化器
        binarizer = MinimapBinarizer(n_clusters=n_clusters)
        
        # 进行二值化
        road_mask = binarizer.Binarize(image)
        obstacle_mask = binarizer.GetObstacleMask(image)
        
        # 可视化聚类结果
        cluster_visual = binarizer.VisualizeClusters(image)
        
        # 显示结果
        print(f"道路标签: {binarizer.GetRoadLabel()}")
        
        # 创建显示图像
        display = np.hstack([
            image,
            cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(obstacle_mask, cv2.COLOR_GRAY2BGR),
            cluster_visual
        ])
        
        # 添加标签
        cv2.putText(display, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Road Mask", (image.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Obstacle Mask", (image.shape[1] * 2 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Clusters (k={n_clusters})", (image.shape[1] * 3 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"KMeans Binarization (k={n_clusters})", display)
        print("按任意键继续下一个测试，按'q'退出...")
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()
    print("\n测试完成")


if __name__ == '__main__':
    main()

