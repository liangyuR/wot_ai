#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图锚点检测器（模板匹配版 + 掩膜支持 + 对比度增强 + 二阶段精修）
增加 ROI 区域，从右下角开始起步，默认取 640x640 像素范围进行匹配。
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class MinimapAnchorDetector:
    def __init__(self, template_path: str, debug: bool = False, multi_scale: bool = False):
        template_path = str(template_path)
        self.template_path_ = template_path
        self.debug_ = debug
        self.multi_scale_ = multi_scale

        tpl_rgba = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if tpl_rgba is None:
            raise FileNotFoundError(f"无法加载模板图像: {template_path}")

        if tpl_rgba.shape[2] == 4:
            self.template_bgr_ = tpl_rgba[..., :3]
            alpha = tpl_rgba[..., 3]
            self.mask_ = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)[1]
            logger.info(f"模板包含透明通道，已生成mask（有效像素: {np.sum(self.mask_>0)}）")
        else:
            self.template_bgr_ = tpl_rgba
            self.mask_ = None
            logger.warning("模板不包含Alpha通道，未启用mask")

        self.template_gray_ = cv2.cvtColor(self.template_bgr_, cv2.COLOR_BGR2GRAY)
        self.tpl_h_, self.tpl_w_ = self.template_gray_.shape[:2]
        self.clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        logger.info(f"模板加载成功: {template_path}, 尺寸=({self.tpl_w_}x{self.tpl_h_})")

    def detect(self, frame: np.ndarray, size: Tuple[int, int] = (640, 640)) -> Optional[Tuple[int, int]]:
        if frame is None or frame.size == 0:
            logger.error("输入帧为空")
            return None

        h, w = frame.shape[:2]
        roi_w, roi_h = 1200, 1200
        start_x = max(0, w - roi_w)
        start_y = max(0, h - roi_h)
        roi = frame[start_y:h, start_x:w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_eq = self.clahe_.apply(gray)

        best_val, best_loc, best_scale = -1.0, None, 1.0
        method = cv2.TM_CCORR_NORMED
        scales = [0.9, 1.0, 1.1] if self.multi_scale_ else [1.0]

        for scale in scales:
            tpl_scaled = cv2.resize(self.template_gray_, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            tpl_scaled_eq = self.clahe_.apply(tpl_scaled)
            mask_scaled = cv2.resize(self.mask_, (tpl_scaled.shape[1], tpl_scaled.shape[0]), interpolation=cv2.INTER_NEAREST) if self.mask_ is not None else None

            res = cv2.matchTemplate(gray_eq, tpl_scaled_eq, method, mask=mask_scaled)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            logger.info(f"scale={scale}, max_val={max_val}")

            if max_val > best_val:
                best_val, best_loc, best_scale = max_val, max_loc, scale

        if best_loc is None:
            logger.warning("模板匹配失败：未找到最大响应")
            return None

        # 二阶段精修（±8像素）
        tpl_w, tpl_h = int(self.tpl_w_ * best_scale), int(self.tpl_h_ * best_scale)
        x0, y0 = best_loc
        delta = 8

        rx0, ry0 = max(0, x0 - delta), max(0, y0 - delta)
        rx1, ry1 = min(roi.shape[1], x0 + tpl_w + delta), min(roi.shape[0], y0 + tpl_h + delta)
        roi_small = gray_eq[ry0:ry1, rx0:rx1]

        tpl_small = cv2.resize(self.template_gray_, (tpl_w, tpl_h), interpolation=cv2.INTER_LINEAR)
        tpl_small = self.clahe_.apply(tpl_small)
        mask_small = cv2.resize(self.mask_, (tpl_w, tpl_h), interpolation=cv2.INTER_NEAREST) if self.mask_ is not None else None

        res2 = cv2.matchTemplate(roi_small, tpl_small, method, mask=mask_small)
        _, max_val2, _, max_loc2 = cv2.minMaxLoc(res2)
        best_val = max_val2
        best_loc = (rx0 + max_loc2[0], ry0 + max_loc2[1])

        # 计算全局坐标
        top_left = (start_x + best_loc[0], start_y + best_loc[1])
        
        # 向右下角偏移约15个像素
        offset_x, offset_y = 12, 12
        top_left = (top_left[0] + offset_x, top_left[1] + offset_y)

        if self.debug_:
            self._draw_debug(frame, top_left, (tpl_w, tpl_h), size, best_val)

        logger.info(f"检测成功: 坐标={top_left}, 置信度={best_val:.3f}, 尺度={best_scale:.2f} (已偏移+{offset_x},+{offset_y})")
        return top_left

    def _draw_debug(self, frame: np.ndarray, top_left: Tuple[int, int], tpl_size: Tuple[int, int], minimap_size: Tuple[int, int], confidence: float):
        x, y = top_left
        tpl_w, tpl_h = tpl_size
        w, h = minimap_size

        debug_img = frame.copy()
        cv2.rectangle(debug_img, (x, y), (x + tpl_w, y + tpl_h), (0, 255, 0), 2)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 255, 0), 1)
        cv2.putText(debug_img, f"Conf: {confidence:.3f}", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # 缩放百分之30图片大小后显示，保持宽高比
        scale_percent = 30  # 30%
        width = int(debug_img.shape[1] * scale_percent / 100)
        height = int(debug_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_debug_img = cv2.resize(debug_img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Minimap Template Match Debug", resized_debug_img)
        cv2.waitKey(1)

    def DetectRegion(self, frame: np.ndarray) -> Optional[dict]:
        """
        检测小地图区域
        
        Args:
            frame: 屏幕帧
        
        Returns:
            小地图区域字典 {x, y, width, height}，失败返回 None
        """
        # 获取屏幕尺寸
        frame_h, frame_w = frame.shape[:2]
        
        # 使用MinimapAnchorDetector检测小地图位置
        top_left = self.anchor_detector_.detect(frame)
        if top_left is None:
            logger.warning("无法检测到小地图位置")
            return None
        
        x, y = top_left
        
        # 根据 top_left 到屏幕右下角的距离自适应计算小地图尺寸
        # 计算可用空间
        available_width = frame_w - x
        available_height = frame_h - y
        
        # 小地图通常是正方形，取宽度和高度的最小值
        minimap_size = min(available_width, available_height)
        
        # 确保尺寸合理（至少大于0，且不超过配置的最大值）
        minimap_size = max(1, min(minimap_size, self.config_.minimap.max_size))
        
        region = {
            'x': x,
            'y': y,
            'width': minimap_size,
            'height': minimap_size
        }
        
        logger.info(f"检测到小地图区域: {region} (自适应尺寸: {minimap_size}x{minimap_size})")
        self.minimap_region_ = region
        return region