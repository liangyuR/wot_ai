#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图掩码提供器：加载离线标注的障碍掩码，并进行透视对齐和膨胀处理
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class MapMaskProvider:
    """地图掩码提供器：管理静态障碍掩码的加载、对齐和膨胀"""
    
    def __init__(self, maps_dir: str, inflate_px: int = 4):
        """
        初始化掩码提供器
        
        Args:
            maps_dir: 地图数据目录路径（包含各地图子目录）
            inflate_px: 膨胀像素数（为车辆留安全边距）
        """
        self.maps_dir_ = Path(maps_dir)
        self.inflate_px_ = max(0, int(inflate_px))
        self.cache_ = {}  # 缓存已加载的掩码和元数据
    
    def _load(self, map_id: str) -> Tuple[np.ndarray, Dict]:
        """
        加载指定地图的掩码和元数据（带缓存）
        
        Args:
            map_id: 地图ID（对应子目录名）
            eg: 胜利之门.png 是原地图，胜利之门_mask.png 是掩码
        
        Returns:
            (mask_1024, meta_dict)
        """
        if map_id in self.cache_:
            return self.cache_[map_id]
        
        mask_path = self.maps_dir_ / map_id / "obstacle_mask_1024.png"
        meta_path = self.maps_dir_ / map_id / "meta.json"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")
        
        # 读取掩码（单通道灰度图）
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码文件: {mask_path}")
        
        # 读取元数据
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        
        # 缓存
        self.cache_[map_id] = (mask, meta)
        return self.cache_[map_id]
    
    @staticmethod
    def _order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
        """
        将四个点排序为：左上、右上、右下、左下
        
        Args:
            pts: 四个点的坐标数组，形状为 (4, 2)
        
        Returns:
            排序后的点数组
        """
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        
        tl = pts[np.argmin(s)]  # 左上：x+y最小
        br = pts[np.argmax(s)]  # 右下：x+y最大
        tr = pts[np.argmin(d)]  # 右上：x-y最小
        bl = pts[np.argmax(d)]  # 左下：x-y最大
        
        return np.float32([tl, tr, br, bl])
    
    def _warp(self, mask_std: np.ndarray, meta: Dict, 
              dst_quad_xy: np.ndarray, out_wh: Tuple[int, int]) -> np.ndarray:
        """
        透视变换：将标准掩码对齐到当前小地图
        
        Args:
            mask_std: 标准尺寸掩码（1024x1024）
            meta: 元数据字典（包含ref_points）
            dst_quad_xy: 目标四角坐标（当前小地图的四角）
            out_wh: 输出尺寸 (width, height)
        
        Returns:
            对齐后的掩码（255=障碍，0=可通行）
        """
        rp = meta["ref_points"]
        src = np.float32([rp["tl"], rp["tr"], rp["br"], rp["bl"]])
        dst = self._order_tl_tr_br_bl(dst_quad_xy.astype(np.float32))
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src, dst)
        
        # 执行透视变换（使用最近邻插值保持二值特性）
        warped = cv2.warpPerspective(
            mask_std, M, out_wh, 
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped
    
    def _inflate(self, mask255: np.ndarray) -> np.ndarray:
        """
        膨胀掩码，为车辆留安全边距
        
        Args:
            mask255: 掩码（255=障碍，0=可通行）
        
        Returns:
            膨胀后的掩码
        """
        if self.inflate_px_ <= 0:
            return mask255
        
        # 使用椭圆核进行膨胀
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (2 * self.inflate_px_ + 1, 2 * self.inflate_px_ + 1)
        )
        return cv2.dilate(mask255, k, iterations=1)
    
    def GetAlignedMask(self, map_id: str, minimap_bgr: np.ndarray, 
                       corners_xy: np.ndarray) -> np.ndarray:
        """
        获取对齐后的障碍掩码（0/1格式）
        
        Args:
            map_id: 地图ID
            minimap_bgr: 当前小地图图像（BGR）
            corners_xy: 小地图四角坐标（形状为(4,2)）
        
        Returns:
            对齐后的掩码（0=可通行，1=障碍），尺寸与minimap_bgr相同
        """
        H, W = minimap_bgr.shape[:2]
        
        # 加载标准掩码和元数据
        mask_std, meta = self._load(map_id)
        
        # 透视变换对齐
        warped255 = self._warp(mask_std, meta, corners_xy, (W, H))
        
        # 膨胀处理
        inflated255 = self._inflate(warped255)
        
        # 转换为0/1格式
        mask01 = (inflated255 > 0).astype(np.uint8)
        
        return mask01

