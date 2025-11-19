#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码处理模块

封装掩码加载、对齐、转栅格的所有逻辑。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from loguru import logger

from src.navigation.core.grid_preprocess import build_inflated_and_cost_map
from src.navigation.config.models import NavigationConfig


@dataclass
class MaskData:
    """掩码数据"""
    aligned_mask: Optional[np.ndarray]  # 对齐后的掩码
    grid: np.ndarray                    # 栅格地图
    cost_map: np.ndarray                # 代价图
    inflated_obstacle: np.ndarray       # 膨胀障碍图


def _load_and_align(
    mask_path: Path,
    minimap_frame: np.ndarray,
    corners_xy: np.ndarray
) -> np.ndarray:
    """
    加载掩码并做透视变换对齐到小地图
    
    Args:
        mask_path: 掩码文件路径
        minimap_frame: 小地图图像（BGR）
        corners_xy: 小地图四角坐标，形状为(4,2)，顺序为 [左上, 右上, 右下, 左下]
    
    Returns:
        对齐后的掩码（0=可通行，1=障碍），尺寸与minimap_frame相同
    """
    # 加载掩码
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩码文件: {mask_path}")
    
    # 获取掩码和小地图尺寸
    mask_h, mask_w = mask.shape[:2]
    minimap_h, minimap_w = minimap_frame.shape[:2]
    
    # 掩码四角坐标（假设掩码是标准矩形）
    src_corners = np.float32([
        [0, 0],           # 左上
        [mask_w, 0],      # 右上
        [mask_w, mask_h], # 右下
        [0, mask_h]       # 左下
    ])
    
    # 目标四角坐标（小地图四角）
    dst_corners = np.float32(corners_xy)
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # 执行透视变换（使用最近邻插值保持二值特性）
    warped = cv2.warpPerspective(
        mask, M, (minimap_w, minimap_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255  # 边界填充为白色（可通行）
    )
    
    # 注意：掩码中白色是可通行区域，黑色是障碍
    mask01 = ((warped < 127).astype(np.uint8))
    return mask01


def _mask_to_grid(mask_01: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    将掩码转换为栅格地图
    
    Args:
        mask_01: 掩码（0=可通行，1=障碍），尺寸为 (H, W)
        grid_size: 栅格尺寸 (width, height)
    
    Returns:
        栅格地图（0=可通行，1=障碍），尺寸为 (height, width)
    """
    grid_h, grid_w = grid_size[1], grid_size[0]
    
    # 使用最近邻插值调整尺寸
    grid = cv2.resize(
        mask_01.astype(np.float32),
        (grid_w, grid_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    return grid.astype(np.uint8)


def load_mask(
    mask_path: Optional[str],
    minimap_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    config: NavigationConfig
) -> MaskData:
    """
    加载掩码（主方法）
    
    Args:
        mask_path: 掩码文件路径，如果为None则创建全可通行地图
        minimap_size: 小地图尺寸 (width, height)
        grid_size: 栅格尺寸 (width, height)
        config: 配置字典
    
    Returns:
        MaskData对象，包含对齐后的掩码、栅格地图、代价图和膨胀障碍图
    """
    try:
        if mask_path and Path(mask_path).exists():
            logger.info(f"加载掩码: {mask_path}")
            
            # 创建一个临时图像用于获取尺寸（实际上只需要尺寸信息）
            minimap_w, minimap_h = minimap_size
            temp_minimap = np.zeros((minimap_h, minimap_w, 3), dtype=np.uint8)
            
            # 假设小地图是矩形，四角坐标为 [左上, 右上, 右下, 左下]
            corners_xy = np.array([
                [0, 0],
                [minimap_w, 0],
                [minimap_w, minimap_h],
                [0, minimap_h]
            ])
            
            aligned_mask = _load_and_align(Path(mask_path), temp_minimap, corners_xy)
            
            # 转换为栅格尺寸
            grid = _mask_to_grid(aligned_mask, grid_size)
            logger.info(f"从掩码构建栅格地图，尺寸: {grid.shape}")
        else:
            # 如果没有掩码，创建一个空的栅格地图（全部可通行）
            logger.warning("未提供掩码，使用全可通行栅格地图")
            grid = np.zeros((grid_size[1], grid_size[0]), dtype=np.uint8)
            aligned_mask = None
        
        # 构建膨胀障碍图和代价图
        if config.mask:
            inflation_radius_px = config.mask.inflation_radius_px
            cost_alpha = config.mask.cost_alpha
        else:
            # 如果没有掩码配置，使用默认值
            inflation_radius_px = 15
            cost_alpha = 10.0
        inflated_obstacle, cost_map = build_inflated_and_cost_map(
            grid,
            inflate_radius_px=inflation_radius_px,
            alpha=cost_alpha
        )
        logger.info(f"障碍膨胀和代价图构建完成: 膨胀半径={inflation_radius_px}px, cost_alpha={cost_alpha}")
        
        return MaskData(
            aligned_mask=aligned_mask,
            grid=grid,
            cost_map=cost_map,
            inflated_obstacle=inflated_obstacle
        )
        
    except Exception as e:
        logger.error(f"掩码加载失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认的全可通行地图
        grid = np.zeros((grid_size[1], grid_size[0]), dtype=np.uint8)
        if config.mask:
            inflation_radius_px = config.mask.inflation_radius_px
            cost_alpha = config.mask.cost_alpha
        else:
            inflation_radius_px = 10
            cost_alpha = 10.0
        inflated_obstacle, cost_map = build_inflated_and_cost_map(
            grid,
            inflate_radius_px=inflation_radius_px,
            alpha=cost_alpha
        )
        return MaskData(
            aligned_mask=None,
            grid=grid,
            cost_map=cost_map,
            inflated_obstacle=inflated_obstacle
        )

