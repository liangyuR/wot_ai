#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测功能测试
测试新的 MinimapDetector（YOLO）
"""

import cv2  
import numpy as np 
from pathlib import Path
from typing import Tuple
from loguru import logger
from wot_ai.game_modules.vision.detection.minimap_detector import MinimapDetector, MinimapDetectionResult
from wot_ai.game_modules.vision.detection.minimap_anchor_detector import MinimapAnchorDetector
from wot_ai.game_modules.navigation.core.path_planner import AStarPlanner


def TestYOLODetector(minimap: np.ndarray, model_path: str, confidence_threshold: float = 0.25) -> Tuple[bool, MinimapDetectionResult, np.ndarray]:
    """测试 YOLO 检测器"""
    try:
        detector = MinimapDetector(model_path, conf_threshold=0.25)
        if not detector.LoadModel():
            logger.error("✗ 模型加载失败")
            return False, MinimapDetectionResult(None, None, []), minimap.copy()
        logger.info("✓ 模型加载成功")
        
        results = detector.Detect(minimap, debug=True)
        logger.info(f"检测结果: {results}")
        return True, results, minimap.copy()
    except Exception as e:
        print(f"✗ YOLO检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, MinimapDetectionResult(None, None, []), minimap.copy()

def LoadMaskAndAlign(mask_path: Path, minimap_frame: np.ndarray, corners_xy: np.ndarray, 
                     erosion_size: int = 3) -> np.ndarray:
    """
    加载掩码并做透视变换对齐到小地图
    
    Args:
        mask_path: 掩码文件路径
        minimap_frame: 小地图图像（BGR）
        corners_xy: 小地图四角坐标，形状为(4,2)，顺序为 [左上, 右上, 右下, 左下]
        erosion_size: 腐蚀操作核大小，用于缩小障碍物，让自身看起来更小
    
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
    
    # 转换为0/1格式（白色=可通行=0，黑色=障碍=1）
    # 注意：掩码中白色是可通行区域，黑色是障碍
    mask01 = ((warped < 127).astype(np.uint8))  # 黑色(<127) = 障碍(1)，白色(>=127) = 可通行(0)
    
    # 对障碍物进行腐蚀操作，缩小障碍物，让自身看起来更小，使狭窄路段可通过
    if erosion_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size * 2 + 1, erosion_size * 2 + 1))
        # 只对障碍物（值为1的区域）进行腐蚀
        mask01 = cv2.erode(mask01, kernel, iterations=1)
        logger.info(f"已对障碍物进行腐蚀操作，核大小: {erosion_size * 2 + 1}")
    
    return mask01


def MaskToGrid(mask_01: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    """
    将掩码转换为栅格地图
    
    Args:
        mask_01: 掩码（0=可通行，1=障碍），尺寸为 (H, W)
        grid_size: 栅格尺寸 (width, height)
    
    Returns:
        栅格地图（0=可通行，1=障碍），尺寸为 (height, width)
    """
    grid_h, grid_w = grid_size[1], grid_size[0]
    mask_h, mask_w = mask_01.shape[:2]
    
    # 使用最近邻插值调整尺寸
    grid = cv2.resize(
        mask_01.astype(np.float32),
        (grid_w, grid_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    return grid.astype(np.uint8)


def WorldToGrid(world_pos: Tuple[float, float], minimap_size: Tuple[int, int], 
                grid_size: Tuple[int, int]) -> Tuple[int, int]:
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
    
    # 限制在栅格范围内
    gx = max(0, min(gx, grid_w - 1))
    gy = max(0, min(gy, grid_h - 1))
    
    return (gx, gy)


def TestMinimapDetection(image_path: str, template_path: str) -> np.ndarray:
    """测试小地图检测功能"""
    # 1. 加载测试图片
    origin_frame = cv2.imread(str(image_path))
    if origin_frame is None:
        logger.error(f"错误：无法读取测试图片: {image_path}")
        return False
    
    h, w = origin_frame.shape[:2]
    logger.info(f"测试图片加载成功: {w}x{h}")

    anchor_detector = MinimapAnchorDetector(template_path=str(template_path), debug=True)
    anchor_pos = anchor_detector.detect(origin_frame)
    if anchor_pos is None:
        logger.error("错误：无法检测到小地图锚点")
        return False
    x, y = anchor_pos
    minimap_region = {
        'x': x,
        'y': y,
        'width': w,
        'height': h
    }
    logger.info(f"小地图锚点: {x}, {y}, 小地图区域: {minimap_region}")
    minimap_frame = origin_frame[minimap_region['y']:minimap_region['y']+minimap_region['height'], minimap_region['x']:minimap_region['x']+minimap_region['width']]
    cv2.imshow("Minimap Frame", minimap_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return minimap_frame

def main():
    """主函数"""
    script_dir = Path(__file__).resolve().parent
    minimap_path = script_dir / "minimap.png"
    if not minimap_path.exists():
        logger.error(f"错误：测试图片不存在: {minimap_path}")
        return False

    minimap_template_path = script_dir / "minimap_border.png"
    if not minimap_template_path.exists():
        logger.error(f"错误：小地图模板不存在: {minimap_template_path}")
        return False

    model_path = script_dir / "best.pt"
    if not model_path.exists():
        logger.error(f"错误：模型不存在: {model_path}")
        return False

    minimap_frame = TestMinimapDetection(minimap_path, minimap_template_path)
    success, results, minimap = TestYOLODetector(minimap_frame, model_path, confidence_threshold=0.75)
    if success:
        logger.info(f"检测结果: {results}")
        logger.info(f"小地图: {minimap}")
    else:
        logger.error("检测失败")
        return False

    # 加载地图掩码: 胜利之门_mask.png
    mask_path = script_dir / "胜利之门_mask.png"
    if not mask_path.exists():
        logger.error(f"错误：掩码文件不存在: {mask_path}")
        return False
    
    logger.info("开始加载地图掩码并规划路径...")
    
    # 获取小地图尺寸
    minimap_h, minimap_w = minimap_frame.shape[:2]
    minimap_size = (minimap_w, minimap_h)
    
    # 假设小地图是矩形，四角坐标为 [左上, 右上, 右下, 左下]
    corners_xy = np.array([
        [0, 0],              # 左上
        [minimap_w, 0],      # 右上
        [minimap_w, minimap_h], # 右下
        [0, minimap_h]       # 左下
    ])
    
    try:
        # 加载并对齐掩码
        aligned_mask = LoadMaskAndAlign(mask_path, minimap_frame, corners_xy)
        logger.info(f"掩码对齐成功，尺寸: {aligned_mask.shape}")
        
        # 构建栅格地图（根据小地图尺寸动态计算，使用256x256或按比例）
        grid_size = (256, 256)  # 固定栅格尺寸
        grid = MaskToGrid(aligned_mask, grid_size)
        logger.info(f"栅格地图构建成功，尺寸: {grid.shape}")
        
        # 检查检测结果
        if results.self_pos is None:
            logger.warning("未检测到自身位置，无法规划路径")
            return success
        
        if results.enemy_flag_pos is None:
            logger.warning("未检测到敌方基地位置，无法规划路径")
            return success
        
        # 将检测结果的位置转换为栅格坐标
        start = WorldToGrid(results.self_pos, minimap_size, grid_size)
        goal = WorldToGrid(results.enemy_flag_pos, minimap_size, grid_size)
        
        logger.info(f"起点（栅格坐标）: {start}")
        logger.info(f"终点（栅格坐标）: {goal}")
        
        # 使用 A* 算法规划路径
        planner = AStarPlanner(enable_smoothing=True, smooth_weight=0.3)
        try:
            path = planner.Plan(grid, start, goal)
            logger.info(f"路径规划成功，路径长度: {len(path)}")
            
            # 可视化路径结果
            vis_img = minimap_frame.copy()
            
            # 绘制栅格地图（缩小显示）
            grid_vis = cv2.resize(grid.astype(np.float32), (minimap_w, minimap_h), 
                                 interpolation=cv2.INTER_NEAREST) * 255
            grid_vis = cv2.cvtColor(grid_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis_img = cv2.addWeighted(vis_img, 0.7, grid_vis, 0.3, 0)
            
            # 绘制路径（将栅格坐标转换回世界坐标）
            if len(path) > 1:
                scale_x = minimap_w / grid_size[0]
                scale_y = minimap_h / grid_size[1]
                
                path_points = []
                for gx, gy in path:
                    wx = int(gx * scale_x)
                    wy = int(gy * scale_y)
                    path_points.append((wx, wy))
                
                # 绘制路径线
                for i in range(len(path_points) - 1):
                    pt1 = path_points[i]
                    pt2 = path_points[i + 1]
                    cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
                
                # 绘制起点和终点
                start_world = path_points[0]
                goal_world = path_points[-1]
                cv2.circle(vis_img, start_world, 5, (0, 255, 0), -1)
                cv2.circle(vis_img, goal_world, 5, (0, 0, 255), -1)
                cv2.putText(vis_img, "Start", (start_world[0] + 10, start_world[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(vis_img, "Goal", (goal_world[0] + 10, goal_world[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imshow("Path Planning Result", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"路径规划失败: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        logger.error(f"掩码处理失败: {e}")
        import traceback
        traceback.print_exc()
        return success

    return success


if __name__ == "__main__":
    main()
