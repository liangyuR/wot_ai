#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图锚点检测器
用于检测小地图左上角在屏幕中的位置，支持可配置的裁剪与调试显示
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from loguru import logger


class MinimapAnchorDetector:
    """
    小地图锚点检测器
    
    通过检测小地图的上边条和左边条颜色来定位小地图左上角位置。
    支持ROI裁剪以提高检测效率，支持调试模式实时显示检测结果。
    
    示例:
        ```python
        detector = MinimapAnchorDetector(
            color_top=(0, 0, 255),  # 上边条颜色 (B, G, R)
            color_left=(0, 0, 255),  # 左边条颜色 (B, G, R)
            tol=20,
            debug=True
        )
        topleft = detector.detect(frame, crop=True)
        ```
    """
    
    def __init__(self, color_top: tuple, color_left: tuple, tol: int = 20, debug: bool = False):
        """
        初始化小地图锚点检测器
        
        Args:
            color_top: 上边条颜色 (B, G, R)
            color_left: 左边条颜色 (B, G, R)
            tol: 颜色容差
            debug: 是否启用调试显示
        """
        self.color_top_bgr_ = np.array(color_top, dtype=np.uint8)
        self.color_left_bgr_ = np.array(color_left, dtype=np.uint8)
        self.tol_ = tol
        self.debug_ = debug
        
        # 将BGR颜色转换为HSV
        color_top_bgr_3d = np.uint8([[[color_top[0], color_top[1], color_top[2]]]])
        color_left_bgr_3d = np.uint8([[[color_left[0], color_left[1], color_left[2]]]])
        
        self.color_top_hsv_ = cv2.cvtColor(color_top_bgr_3d, cv2.COLOR_BGR2HSV)[0][0]
        self.color_left_hsv_ = cv2.cvtColor(color_left_bgr_3d, cv2.COLOR_BGR2HSV)[0][0]
        
        logger.info(f"MinimapAnchorDetector 初始化: color_top={color_top}, color_left={color_left}, tol={tol}, debug={debug}")
    
    def detect(self, frame: np.ndarray, crop: bool = True, region: Optional[tuple] = None, 
               size: Optional[tuple] = (640, 640)) -> Optional[tuple]:
        """
        检测小地图左上角坐标
        
        Args:
            frame: 输入的屏幕帧（BGR格式）
            crop: 是否裁剪右下角区域
            region: 指定裁剪区域 (x, y, w, h)，若为 None 则自动使用屏幕右下角
            size: 小地图的固定尺寸 (w, h)，用于绘制与验证
        
        Returns:
            (x, y): 小地图左上角在屏幕中的坐标，若检测失败返回 None
        """
        if frame is None or frame.size == 0:
            logger.error("输入帧为空")
            return None
        
        H, W = frame.shape[:2]
        
        # ROI提取
        if crop:
            if region is not None:
                # 使用指定的裁剪区域
                x, y, w, h = region
                roi = frame[y:y+h, x:x+w]
                offset = (x, y)
            else:
                # 默认提取屏幕右下角一半区域
                roi = frame[int(H*0.5):, int(W*0.5):]
                offset = (int(W*0.5), int(H*0.5))
        else:
            # 不裁剪，使用整个屏幕
            roi = frame
            offset = (0, 0)
        
        if roi.size == 0:
            logger.error("ROI区域为空")
            return None
        
        # 转换至HSV空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 检测上边条和左边条
        top_y = self._find_color_band(hsv, self.color_top_hsv_, self.tol_, 'horizontal')
        left_x = self._find_color_band(hsv, self.color_left_hsv_, self.tol_, 'vertical')
        
        if top_y is None or left_x is None:
            if self.debug_:
                logger.debug(f"检测失败: top_y={top_y}, left_x={left_x}")
            return None
        
        # 计算左上角在原始屏幕中的坐标
        top_left = (offset[0] + left_x, offset[1] + top_y)
        
        # 调试绘制
        if self.debug_:
            self._draw_debug(frame, top_left, size)
        
        logger.debug(f"检测到小地图左上角: {top_left}")
        return top_left
    
    def _find_color_band(self, hsv: np.ndarray, color: tuple, tol: int, axis: str) -> Optional[int]:
        """
        内部函数：在HSV图像中查找指定颜色的带状区域
        
        Args:
            hsv: 输入HSV图像
            color: 目标颜色HSV (H, S, V)
            tol: 容差
            axis: 'horizontal' 或 'vertical'
        
        Returns:
            匹配到的行或列索引，若未找到返回 None
        """
        # 创建颜色范围
        lower = np.array([
            max(0, color[0] - tol),
            max(0, color[1] - tol),
            max(0, color[2] - tol)
        ], dtype=np.uint8)
        
        upper = np.array([
            min(180, color[0] + tol),
            min(255, color[1] + tol),
            min(255, color[2] + tol)
        ], dtype=np.uint8)
        
        # 创建掩膜
        mask = cv2.inRange(hsv, lower, upper)
        
        # 根据方向求和
        if axis == 'horizontal':
            # 水平方向：沿列求和，找到首次出现像素密度超过阈值的行
            sums = np.sum(mask, axis=1)
        elif axis == 'vertical':
            # 垂直方向：沿行求和，找到首次出现像素密度超过阈值的列
            sums = np.sum(mask, axis=0)
        else:
            logger.error(f"无效的axis参数: {axis}")
            return None
        
        # 计算阈值：至少需要一定比例的像素匹配
        threshold = sums.shape[0] * 0.3  # 至少30%的像素匹配
        
        # 查找首次超过阈值的索引
        indices = np.where(sums > threshold)[0]
        if len(indices) > 0:
            return int(indices[0])
        
        return None
    
    def _draw_debug(self, frame: np.ndarray, topleft: tuple, size: tuple) -> None:
        """
        调试模式下绘制边框
        
        Args:
            frame: 原始屏幕帧
            topleft: 左上角坐标 (x, y)
            size: 小地图宽高 (w, h)
        """
        if frame is None or topleft is None or size is None:
            return
        
        x, y = topleft
        w, h = size
        
        # 绘制矩形框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 绘制左上角标记
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        # 显示调试窗口
        cv2.imshow("Minimap Anchor Debug", frame)
        cv2.waitKey(1)

