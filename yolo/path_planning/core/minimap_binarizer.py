#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图二值化模块：使用KMeans聚类进行地图二值化
识别可通行区域和障碍物区域
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from sklearn.cluster import KMeans


class MinimapBinarizer:
    """小地图二值化器：使用KMeans聚类识别可通行区域"""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        初始化二值化器
        
        Args:
            n_clusters: KMeans聚类数量（默认5）
            random_state: 随机种子（默认42，保证结果可复现）
        """
        self.n_clusters_ = n_clusters
        self.random_state_ = random_state
        self.kmeans_model_ = None
        self.binary_mask_ = None
        self.road_label_ = None
    
    def Binarize(self, image: np.ndarray) -> np.ndarray:
        """
        对地图图像进行二值化
        
        Args:
            image: 输入图像（BGR格式）
        
        Returns:
            二值化掩码（255=可通行区域，0=障碍物）
        """
        if image is None or image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        
        # 1. 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape
        pixels = hsv.reshape(-1, 3)
        
        # 2. KMeans聚类
        self.kmeans_model_ = KMeans(
            n_clusters=self.n_clusters_,
            n_init=10,
            random_state=self.random_state_,
            verbose=0
        )
        self.kmeans_model_.fit(pixels)
        labels = self.kmeans_model_.labels_.reshape(h, w)
        centers = self.kmeans_model_.cluster_centers_
        
        # 3. 自动找出"亮 + 饱和度低"的类（可通行区域）
        brightness = centers[:, 2]   # V 分量（亮度）
        saturation = centers[:, 1]   # S 分量（饱和度）
        score = brightness - 0.5 * saturation  # 可通行区域一般亮且不饱和
        self.road_label_ = np.argmax(score)
        
        # 4. 生成二值mask（可通行区域=255，障碍物=0）
        mask = (labels == self.road_label_).astype(np.uint8) * 255
        
        # 反转mask：当前识别结果相反，需要反转
        mask = 255 - mask
        
        self.binary_mask_ = mask
        return mask
    
    def GetObstacleMask(self, image: np.ndarray) -> np.ndarray:
        """
        获取障碍物掩码（障碍物=255，可通行=0）
        
        Args:
            image: 输入图像（BGR格式）
        
        Returns:
            障碍物掩码
        """
        road_mask = self.Binarize(image)
        return 255 - road_mask  # 反转：障碍物=255，可通行=0
    
    def GetBinaryMask(self) -> Optional[np.ndarray]:
        """
        获取上次二值化的结果
        
        Returns:
            二值化掩码或None
        """
        return self.binary_mask_
    
    def GetRoadLabel(self) -> Optional[int]:
        """
        获取识别为道路的聚类标签
        
        Returns:
            道路标签或None
        """
        return self.road_label_
    
    def VisualizeClusters(self, image: np.ndarray) -> np.ndarray:
        """
        可视化聚类结果（每个聚类用不同颜色显示）
        
        Args:
            image: 输入图像（BGR格式）
        
        Returns:
            可视化结果图像
        """
        if image is None or image.size == 0:
            return image
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape
        pixels = hsv.reshape(-1, 3)
        
        # 如果没有模型，先进行聚类
        if self.kmeans_model_ is None:
            self.Binarize(image)
        
        labels = self.kmeans_model_.labels_.reshape(h, w)
        centers = self.kmeans_model_.cluster_centers_
        
        # 创建可视化图像：每个像素用其聚类中心的颜色
        visual = np.zeros_like(image)
        for label in range(self.n_clusters_):
            mask = (labels == label)
            # 将HSV中心转换回BGR用于显示
            center_hsv = centers[label].reshape(1, 1, 3).astype(np.uint8)
            center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            visual[mask] = center_bgr
        
        return visual

