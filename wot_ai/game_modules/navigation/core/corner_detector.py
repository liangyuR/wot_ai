#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图四角检测器：通过边缘检测和轮廓分析检测小地图的四个角点
"""

import cv2
import numpy as np
from typing import Optional


def detect_minimap_corners(minimap_bgr: np.ndarray) -> np.ndarray:
    """
    检测小地图的四个角点（左上、右上、右下、左下）
    
    Args:
        minimap_bgr: 小地图图像（BGR格式）
    
    Returns:
        四个角点坐标，形状为 (4, 2)，顺序为 [tl, tr, br, bl]
        如果检测失败，返回图像四角作为回退
    """
    H, W = minimap_bgr.shape[:2]
    
    # 转换为灰度图
    gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
    
    # Canny边缘检测
    edges = cv2.Canny(gray, 40, 120)
    
    # 膨胀边缘，连接断开的线段
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 回退：返回图像四角
        return np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]])
    
    # 找到最大轮廓
    c = max(contours, key=cv2.contourArea)
    
    # 获取最小外接矩形
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    
    # 排序角点：左上、右上、右下、左下
    s = box.sum(axis=1)
    d = np.diff(box, axis=1).ravel()
    
    tl = box[np.argmin(s)]  # 左上：x+y最小
    br = box[np.argmax(s)]  # 右下：x+y最大
    tr = box[np.argmin(d)]  # 右上：x-y最小
    bl = box[np.argmax(d)]  # 左下：x-y最大
    
    return np.float32([tl, tr, br, bl])

