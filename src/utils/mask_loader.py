#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码处理模块

封装掩码加载、对齐、转栅格的所有逻辑。
"""

import numpy as np
import cv2

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


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
    mask = _imread_gray(mask_path)
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
        borderValue=0  # 边界填充为黑色（不可通行）
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

def _imread_gray(path: Path) -> np.ndarray:
    """兼容中文路径的灰度图读取"""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"文件不存在: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取掩码文件: {path}")
    return img

def load_mask(
    mask_path: Optional[str],
    minimap_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    inflation_radius_px: int = 15,
    cost_alpha: float = 10.0,
    soft_obstacle_cost: float = 500.0,
) -> MaskData:
    """加载掩码（无 NavigationConfig 依赖版）

    Args:
        mask_path: 掩码文件路径，如果为 None 或不存在则创建全可通行地图
        minimap_size: 小地图尺寸 (width, height)
        grid_size: 栅格尺寸 (width, height)
        inflation_radius_px: 障碍膨胀半径（像素）
        cost_alpha: 靠近障碍的额外代价权重
        soft_obstacle_cost: 软障碍（膨胀区域）的代价，默认 500.0

    Returns:
        MaskData 对象，包含对齐后的掩码、栅格地图、代价图和膨胀障碍图
    """

    aligned_mask = None
    minimap_w, minimap_h = minimap_size
    assert minimap_w > 0 and minimap_h > 0, "minimap_size 必须为正数"
    grid_w, grid_h = grid_size
    assert grid_w > 0 and grid_h > 0, "grid_size 必须为正数"

    try:
        # --------------------------------------------------------
        # 1. 加载 / 对齐掩码 → 栅格地图
        # --------------------------------------------------------
        if mask_path and Path(mask_path).exists():
            logger.info(f"加载掩码: {mask_path}")

            # 这里只是为了传给 _load_and_align 尺寸信息
            temp_minimap = np.zeros((minimap_h, minimap_w, 3), dtype=np.uint8)

            # 小地图矩形四角坐标 [左上, 右上, 右下, 左下]
            corners_xy = np.array([
                [0, 0],
                [minimap_w, 0],
                [minimap_w, minimap_h],
                [0, minimap_h],
            ])

            aligned_mask = _load_and_align(Path(mask_path), temp_minimap, corners_xy)

            # 转换为栅格尺寸
            grid = _mask_to_grid(aligned_mask, (grid_w, grid_h))
            logger.info(f"从掩码构建栅格地图，尺寸: {grid.shape}")
        else:
            # 没有掩码文件 → 全可通行
            logger.warning("未提供有效掩码，使用全可通行栅格地图")
            grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    except Exception as e:
        logger.error(f"掩码加载或转换失败，回退到全可通行地图: {e}")
        import traceback

        traceback.print_exc()
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        aligned_mask = None

    # ------------------------------------------------------------
    # 2. 构建膨胀障碍图和代价图（单一出口逻辑）
    # ------------------------------------------------------------
    inflated_obstacle, cost_map = build_inflated_and_cost_map(
        grid,
        inflate_radius_px=inflation_radius_px,
        alpha=cost_alpha,
        soft_obstacle_cost=soft_obstacle_cost,
    )
    logger.info(
        "障碍膨胀和代价图构建完成: 膨胀半径=%d px, cost_alpha=%.2f, soft_obstacle_cost=%.1f",
        inflation_radius_px,
        cost_alpha,
        soft_obstacle_cost,
    )

    return MaskData(
        aligned_mask=aligned_mask,
        grid=grid,
        cost_map=cost_map,
        inflated_obstacle=inflated_obstacle,
    )


def build_inflated_and_cost_map(
    obstacle_map: np.ndarray,
    inflate_radius_px: int,
    alpha: float,
    soft_obstacle_cost: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建膨胀障碍图和代价图(cost map)。

    Args:
        obstacle_map: 0/1 二值图, 1 表示障碍
        inflate_radius_px: 障碍膨胀半径（像素）
        alpha: 靠近障碍的额外代价权重
        soft_obstacle_cost: 软障碍（膨胀区域）的代价，默认 500.0

    Returns:
        inflated_obstacle: 0/1 二值图, 1 表示障碍（已膨胀）
        cost_map: float32, free 区域的移动代价（>=1），硬障碍为 np.inf，软障碍为 soft_obstacle_cost
    """
    assert obstacle_map.dtype == np.uint8, "建议用 uint8 二值图 (0/1 或 0/255)"

    # 统一成 0/1
    obs = (obstacle_map > 0).astype(np.uint8)

    # 1) 障碍膨胀（使用椭圆核，避免路径出现不自然的边角）
    k = 2 * inflate_radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inflated = cv2.dilate(obs, kernel, iterations=1)
    
    logger.info(f"障碍膨胀完成: 膨胀半径={inflate_radius_px}px, 椭圆核大小={k}x{k}")

    # 2) 距离变换（对 free 区域算离障碍的距离）
    # 注意：这里使用膨胀后的障碍来计算距离，用于中心化代价
    free = (inflated == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)  # float32

    # 归一化，越靠近障碍 cost 越大
    if dist.max() > 0:
        dist_norm = dist / (dist.max() + 1e-6)
    else:
        dist_norm = dist

    # 基础代价 = 1，越远离障碍代价越接近 1，越靠近障碍代价越大
    # 这会让路径更倾向于走道路中央
    cost_map = 1.0 + alpha * (1.0 - dist_norm)
    cost_map = cost_map.astype(np.float32, copy=False)

    # 3) 区分硬障碍和软障碍
    # 硬障碍（原始障碍）→ np.inf（绝对不可穿越）
    cost_map[obs != 0] = np.inf
    
    # 软障碍（膨胀区域，非原始障碍）→ 有限大值（紧急情况可穿越，但代价极高）
    inflated_only = (inflated != 0) & (obs == 0)
    cost_map[inflated_only] = soft_obstacle_cost
    
    logger.info(
        f"障碍分类完成: 硬障碍={np.sum(obs != 0)}个格子(inf), "
        f"软障碍={np.sum(inflated_only)}个格子({soft_obstacle_cost})"
    )

    finite = np.isfinite(cost_map)
    if finite.any():
        max_cost = float(cost_map[finite].max())
    else:
        max_cost = float("nan")

    logger.info(f"代价图构建完成: alpha={alpha}, soft_obstacle_cost={soft_obstacle_cost}, 最大有限代价={max_cost:.2f}")

    return inflated, cost_map

def _test_mask_pipeline_basic(
    mask_file: str,
    minimap_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    inflation_radius_px: int,
    cost_alpha: float,
):
    """简单单元测试：用真实掩码文件验证 load_mask + cost_map 流程"""

    from pathlib import Path

    mask_path = Path(mask_file)
    assert mask_path.exists(), f"掩码文件不存在: {mask_path}"

    logger.info(f"使用真实掩码文件进行测试: {mask_path}")
    logger.debug(f"传入参数：mask_path={mask_path}, minimap_size={minimap_size}, grid_size={grid_size}, inflation_radius_px={inflation_radius_px}, cost_alpha={cost_alpha}")

    # 1. 跑完整的 load_mask 流程
    md = load_mask(
        mask_path=str(mask_path),
        minimap_size=minimap_size,
        grid_size=grid_size,
        inflation_radius_px=inflation_radius_px,
        cost_alpha=cost_alpha,
    )

    logger.debug(f"load_mask 结果：grid shape={md.grid.shape}, inflated_obstacle shape={md.inflated_obstacle.shape}, cost_map shape={md.cost_map.shape}")

    # 2. 基础形状检查
    assert md.grid.shape == (grid_size[1], grid_size[0]), f"grid 尺寸不匹配: {md.grid.shape} vs {grid_size}"
    assert md.inflated_obstacle.shape == md.grid.shape, f"inflated_obstacle 尺寸不匹配: {md.inflated_obstacle.shape} vs {md.grid.shape}"
    assert md.cost_map.shape == md.grid.shape, f"cost_map 尺寸不匹配: {md.cost_map.shape} vs {md.grid.shape}"

    # 3. grid 只允许 0/1（宽松一点，只要是 0/1 就行）
    unique_vals = np.unique(md.grid)
    logger.debug(f"grid 唯一值: {unique_vals}")
    assert set(unique_vals.tolist()).issubset({0, 1}), f"grid 非二值: {unique_vals}"

    # 4. 障碍/代价基本 sanity check（允许没有障碍的极端情况）
    inflated = (md.inflated_obstacle > 0)
    free_mask = (inflated == 0)
    obs_mask = (inflated != 0)
    logger.debug(f"grid 总像素: {md.grid.size}, inflated_obstacle 总障碍像素: {obs_mask.sum()}, free 区域像素: {free_mask.sum()}")

    if obs_mask.any():
        logger.debug(f"障碍区域像素数量: {obs_mask.sum()}")
        assert np.isinf(md.cost_map[obs_mask]).all(), "障碍区域 cost 应为 inf"

    if free_mask.any():
        logger.debug(f"free 区域像素数量: {free_mask.sum()}")
        assert np.isfinite(md.cost_map[free_mask]).all(), "free 区域 cost 应为有限值"
        min_free_cost = float(md.cost_map[free_mask].min())
        max_free_cost = float(md.cost_map[free_mask].max())
        logger.debug(f"free 区域最小 cost: {min_free_cost}, 最大 cost: {max_free_cost}")
        assert min_free_cost >= 1.0, f"free 区域最小代价应 >= 1，当前为 {min_free_cost}"

    # 5. 可视化：对齐掩码 / grid / cost_map
    try:
        # 对齐后的掩码
        if md.aligned_mask is not None:
            aligned_vis = (md.aligned_mask * 255).astype(np.uint8)
            cv2.imwrite("./aligned_mask_vis.png", aligned_vis)
            logger.info("aligned_mask 可视化已保存: ./aligned_mask_vis.png")
            logger.debug(f"aligned_mask 可视化 shape: {aligned_vis.shape}, dtype: {aligned_vis.dtype}, unique: {np.unique(aligned_vis)}")

        # 栅格地图
        grid_vis = (md.grid * 255).astype(np.uint8)
        cv2.imwrite("./grid_vis.png", grid_vis)
        logger.info("grid 可视化已保存: ./grid_vis.png")
        logger.debug(f"grid_vis shape: {grid_vis.shape}, dtype: {grid_vis.dtype}, unique: {np.unique(grid_vis)}")

        # 成本图
        cost_vis = md.cost_map.copy().astype(np.float32)
        finite_mask = np.isfinite(cost_vis)
        if finite_mask.any():
            mn, mx = cost_vis[finite_mask].min(), cost_vis[finite_mask].max()
            logger.debug(f"cost_map finite min: {mn}, max: {mx}")
            cost_vis[finite_mask] = (
                255 * (cost_vis[finite_mask] - mn) / (mx - mn + 1e-6)
            )
        cost_vis[~finite_mask] = 255  # 障碍设为白色

        cost_vis = cost_vis.astype(np.uint8)
        cv2.imwrite("./cost_map_vis.png", cost_vis)
        logger.info("cost_map 可视化已保存: ./cost_map_vis.png")
        logger.debug(f"cost_map_vis shape: {cost_vis.shape}, dtype: {cost_vis.dtype}, unique: {np.unique(cost_vis)}")

    except Exception as e:
        logger.warning(f"可视化生成失败: {e}")

    logger.info("mask_utils 基本单元测试通过（真实掩码）")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load a mask file and process it with load_mask.")
    parser.add_argument("mask_path", type=str, help="Path to the mask image file (ignored, test uses synthetic mask)")
    parser.add_argument("--minimap_size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=[64, 64], help="Minimap size as width height (default: 64 64)")
    parser.add_argument("--grid_size", type=int, nargs=2, metavar=('COLS', 'ROWS'), default=[16, 16], help="Grid size as cols rows (default: 16 16)")
    parser.add_argument("--inflation_radius_px", type=int, default=3, help="Obstacle inflation radius in pixels (default: 3)")
    parser.add_argument("--cost_alpha", type=float, default=5.0, help="Free region cost scaling factor (default: 5.0)")
    args = parser.parse_args()

    _test_mask_pipeline_basic(args.mask_path, tuple(args.minimap_size), tuple(args.grid_size), args.inflation_radius_px, args.cost_alpha)
