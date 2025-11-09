#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地形分析模块：从小地图图像中提取障碍与通行区域
"""

import cv2
import numpy as np

class TerrainAnalyzer:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def Analyze(self, minimap: np.ndarray) -> np.ndarray:
        """
        检测小地图中不可通行区域。
        Args:
            minimap: 小地图图像 (BGR)
        Returns:
            mask: 二值图，1 = 不可通行区域，0 = 可通行区域
        """
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

        # 阈值分割低亮度边缘（通常是障碍边界）
        _, dark_mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)

        # 形态学操作，强化边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        edges = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 填充闭合区域（估算障碍块）
        filled = edges.copy()
        h, w = edges.shape
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(filled, flood_mask, (0,0), 255)
        filled_inv = cv2.bitwise_not(filled)

        # 合并边界和内部区域
        obstacle_mask = cv2.bitwise_or(edges, filled_inv)

        # 二值化为0/1
        binary = (obstacle_mask > 0).astype(np.uint8)

        if self.debug:
            cv2.imshow("edges", edges)
            cv2.imshow("obstacle_mask", obstacle_mask)
            cv2.waitKey(1)

        return binary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python terrain_analyzer.py <minimap_image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    minimap = cv2.imread(image_path)
    if minimap is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    analyzer = TerrainAnalyzer(debug=True)
    obstacle_mask = analyzer.Analyze(minimap)

    cv2.imshow("Original Minimap", minimap)
    cv2.imshow("Obstacle Mask", obstacle_mask * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
\