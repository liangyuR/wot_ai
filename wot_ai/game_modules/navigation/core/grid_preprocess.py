#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
栅格预处理模块：障碍膨胀和代价图构建

功能：
- 对障碍进行膨胀，为车辆预留安全距离
- 构建距离代价图，引导路径远离障碍
"""

import cv2
import numpy as np
from typing import Tuple
from loguru import logger


def build_inflated_and_cost_map(
    obstacle_map: np.ndarray,
    inflate_radius_px: int = 10,
    alpha: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建膨胀障碍图和代价图(cost map)。

    Args:
        obstacle_map: 0/1 二值图, 1 表示障碍
        inflate_radius_px: 障碍膨胀半径（像素）
        alpha: 靠近障碍的额外代价权重

    Returns:
        inflated_obstacle: 0/1 二值图, 1 表示障碍（已膨胀）
        cost_map: float32, free 区域的移动代价（>=1），障碍处为 np.inf
    """
    assert obstacle_map.dtype == np.uint8, "建议用 uint8 二值图 (0/1 或 0/255)"

    # 统一成 0/1
    obs = (obstacle_map > 0).astype(np.uint8)

    # 1) 障碍膨胀
    k = 2 * inflate_radius_px + 1
    kernel = np.ones((k, k), np.uint8)
    inflated = cv2.dilate(obs, kernel, iterations=1)
    
    logger.debug(f"障碍膨胀完成: 膨胀半径={inflate_radius_px}px, 核大小={k}x{k}")

    # 2) 距离变换（对 free 区域算离障碍的距离）
    free = (inflated == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)  # float32

    # 归一化，越靠近障碍 cost 越大
    if dist.max() > 0:
        dist_norm = dist / (dist.max() + 1e-6)
    else:
        dist_norm = dist

    # 基础代价 = 1，越远离障碍代价越接近 1，越靠近障碍代价越大
    cost_map = 1.0 + alpha * (1.0 - dist_norm)

    # 障碍处不可通行
    cost_map[inflated != 0] = np.inf

    logger.debug(f"代价图构建完成: alpha={alpha}, 最大代价={cost_map[cost_map != np.inf].max():.2f}")

    return inflated, cost_map

