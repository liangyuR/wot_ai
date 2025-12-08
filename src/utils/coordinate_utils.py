#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换工具模块

提供世界坐标和栅格坐标之间的转换功能。
"""

from typing import Tuple


def world_to_grid(
    world_pos: Tuple[float, float],
    minimap_size: Tuple[int, int],
    grid_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    将世界坐标（小地图像素坐标）转换为栅格坐标
    
    Args:
        world_pos: 世界坐标 (x, y)
        minimap_size: 小地图尺寸 (width, height)
        grid_size: 栅格尺寸 (width, height)
    
    Returns:
        栅格坐标 (x, y)
    """
    wx, wy = world_pos
    minimap_w, minimap_h = minimap_size
    grid_w, grid_h = grid_size
    
    # 计算缩放比例
    scale_x = grid_w / minimap_w
    scale_y = grid_h / minimap_h
    
    # 转换坐标
    gx = int(wx * scale_x)
    gy = int(wy * scale_y)
    
    # 确保坐标在有效范围内
    gx = max(0, min(gx, grid_w - 1))
    gy = max(0, min(gy, grid_h - 1))
    
    return (gx, gy)


def grid_to_world(
    grid_pos: Tuple[int, int],
    minimap_size: Tuple[int, int],
    grid_size: Tuple[int, int]
) -> Tuple[float, float]:
    """
    将栅格坐标转换为世界坐标（小地图像素坐标）
    
    Args:
        grid_pos: 栅格坐标 (x, y)
        minimap_size: 小地图尺寸 (width, height)
        grid_size: 栅格尺寸 (width, height)
    
    Returns:
        世界坐标 (x, y)
    """
    gx, gy = grid_pos
    minimap_w, minimap_h = minimap_size
    grid_w, grid_h = grid_size
    
    # 计算缩放比例
    scale_x = minimap_w / grid_w
    scale_y = minimap_h / grid_h
    
    # 转换坐标
    wx = gx * scale_x
    wy = gy * scale_y
    
    return (wx, wy)

