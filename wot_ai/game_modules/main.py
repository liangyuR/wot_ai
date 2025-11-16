#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序：整合小地图检测、路径规划和自动控制功能

功能流程：
1. 启动main.py
2. 识别小地图位置（MinimapAnchorDetector）
3. YOLO检测小地图并输出结果（MinimapDetector）
4. 规划路径（AStarPlanner + 掩码加载）
5. 在小地图上层覆盖UI显示规划结果（TransparentOverlay）
6. 根据规划结果操作WASD控制坦克（ControlService + NavigationExecutor）
"""

import sys
import time
import math
import threading
import queue
from pathlib import Path
from typing import Optional, Dict
import signal

import numpy as np
import cv2
from loguru import logger

# 导入所需模块
from wot_ai.game_modules.navigation.service.capture_service import CaptureService
from wot_ai.game_modules.vision.detection.minimap_anchor_detector import MinimapAnchorDetector
from wot_ai.game_modules.vision.detection.minimap_detector import MinimapDetector, MinimapDetectionResult
from wot_ai.game_modules.navigation.core.path_planner import AStarPlanner
from wot_ai.game_modules.navigation.core.grid_preprocess import build_inflated_and_cost_map
from wot_ai.game_modules.navigation.core.planner_astar import astar_with_cost
from wot_ai.game_modules.navigation.core.path_smoothing import smooth_path_los
from wot_ai.game_modules.navigation.ui.transparent_overlay import TransparentOverlay
from wot_ai.game_modules.navigation.service.control_service import ControlService
from wot_ai.game_modules.navigation.core.navigation_executor import NavigationExecutor


class NavigationMain:
    """导航主程序"""
    
    def __init__(self, config: Dict):
        """
        初始化导航主程序
        
        Args:
            config: 配置字典，包含：
                - model_path: YOLO模型路径
                - template_path: 小地图模板路径
                - minimap_size: 小地图尺寸 (width, height)
                - mask_path: 掩码路径（可选）
                - grid_size: 栅格尺寸 (width, height)，默认 (256, 256)
                - erosion_size: 腐蚀操作核大小，默认 3
        """
        self.config_ = config
        self.running_ = False
        
        # 服务实例
        self.capture_service_: Optional[CaptureService] = None
        self.anchor_detector_: Optional[MinimapAnchorDetector] = None
        self.minimap_detector_: Optional[MinimapDetector] = None
        self.planner_: Optional[AStarPlanner] = None
        self.overlay_: Optional[TransparentOverlay] = None
        self.control_service_: Optional[ControlService] = None
        self.nav_executor_: Optional[NavigationExecutor] = None
        
        # 小地图区域信息
        self.minimap_region_: Optional[Dict] = None
        
        # 当前状态（仅用于初始化，后续由各线程自己维护）
        self.current_grid_: Optional[np.ndarray] = None
        self.current_aligned_mask_: Optional[np.ndarray] = None  # 对齐后的掩码（用于UI显示）
        self.current_cost_map_: Optional[np.ndarray] = None  # 代价图（用于路径规划）
        self.current_inflated_obstacle_: Optional[np.ndarray] = None  # 膨胀后的障碍图（用于LOS平滑）
        
        # 使用队列传递数据（完全无锁）
        self.detection_queue_ = queue.Queue(maxsize=10)  # 支持30FPS
        self.path_queue_ = queue.Queue(maxsize=1)  # 只保留最新路径
        
        # 线程对象
        self.detection_thread_: Optional[threading.Thread] = None
        self.control_thread_: Optional[threading.Thread] = None
        self.ui_thread_: Optional[threading.Thread] = None
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._SignalHandler_)
        signal.signal(signal.SIGTERM, self._SignalHandler_)
    
    def _SignalHandler_(self, signum, frame):
        """信号处理器（Ctrl+C）"""
        logger.info("收到退出信号，正在关闭...")
        self.Stop()
        sys.exit(0)
    
    def Initialize(self) -> bool:
        """
        初始化所有服务
        
        Returns:
            是否初始化成功
        """
        try:
            logger.info("开始初始化服务...")
            
            # 1. 初始化屏幕捕获服务
            logger.info("初始化屏幕捕获服务...")
            self.capture_service_ = CaptureService(monitor_index=1)
            
            # 2. 初始化小地图锚点检测器
            template_path = self.config_.get('template_path')
            if not template_path or not Path(template_path).exists():
                logger.error(f"小地图模板不存在: {template_path}")
                return False
            
            logger.info(f"初始化小地图锚点检测器: {template_path}")
            self.anchor_detector_ = MinimapAnchorDetector(
                template_path=template_path,
                debug=False,
                multi_scale=False
            )
            
            # 3. 初始化YOLO检测器
            model_path = self.config_.get('model_path')
            if not model_path or not Path(model_path).exists():
                logger.error(f"YOLO模型不存在: {model_path}")
                return False
            
            logger.info(f"初始化YOLO检测器: {model_path}")
            # 读取输入尺寸配置（如果指定）
            input_size_config = self.config_.get('input_size', None)
            input_size = None
            if input_size_config:
                if isinstance(input_size_config, (list, tuple)) and len(input_size_config) == 2:
                    input_size = tuple(input_size_config)
                else:
                    logger.warning("input_size配置格式错误，应为(width, height)，忽略")
            
            self.minimap_detector_ = MinimapDetector(
                model_path=model_path,
                conf_threshold=self.config_.get('conf_threshold', 0.25),
                iou_threshold=self.config_.get('iou_threshold', 0.75),
            )
            
            if not self.minimap_detector_.LoadModel():
                logger.error("YOLO模型加载失败")
                return False
            
            # 4. 初始化路径规划器
            logger.info("初始化路径规划器...")
            self.planner_ = AStarPlanner(
                enable_smoothing=self.config_.get('enable_smoothing', True),
                smooth_weight=self.config_.get('smooth_weight', 0.3)
            )
            
            # 5. 初始化控制服务
            logger.info("初始化控制服务...")
            self.control_service_ = ControlService()
            
            # 6. 初始化导航执行器
            logger.info("初始化导航执行器...")
            self.nav_executor_ = NavigationExecutor(
                control_service=self.control_service_,
                move_speed=self.config_.get('move_speed', 1.0),
                rotation_smooth=self.config_.get('rotation_smooth', 0.3)
            )
            
            # 7. TransparentOverlay 将在检测到小地图位置后初始化
            # 因为需要知道小地图的确切位置和尺寸
            
            logger.info("所有服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def DetectMinimapRegion(self, frame: np.ndarray) -> Optional[Dict]:
        """
        检测小地图区域
        
        Args:
            frame: 屏幕帧
        
        Returns:
            小地图区域字典 {x, y, width, height}，失败返回 None
        """
        # 获取屏幕尺寸
        frame_h, frame_w = frame.shape[:2]
        
        # 使用默认尺寸进行模板匹配（用于检测位置）
        default_size = self.config_.get('minimap_size', (640, 640))
        
        # 使用MinimapAnchorDetector检测小地图位置
        top_left = self.anchor_detector_.detect(frame, size=default_size)
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
        max_size = self.config_.get('max_minimap_size', 1000)
        minimap_size = max(1, min(minimap_size, max_size))
        
        region = {
            'x': x,
            'y': y,
            'width': minimap_size,
            'height': minimap_size
        }
        
        logger.info(f"检测到小地图区域: {region} (自适应尺寸: {minimap_size}x{minimap_size})")
        return region
    
    def InitializeOverlay(self, minimap_region: Dict) -> bool:
        """
        初始化透明覆盖层（显示在左上角，避免影响 mss 录制小地图）
        
        Args:
            minimap_region: 小地图区域 {x, y, width, height}（用于确定 overlay 尺寸）
        
        Returns:
            是否初始化成功
        """
        try:
            # overlay 显示在左上角，尺寸与小地图相同以保持路径显示比例
            overlay_width = minimap_region['width']
            overlay_height = minimap_region['height']
            logger.info(f"初始化透明覆盖层，位置: (0, 0)（左上角）, "
                       f"尺寸: {overlay_width}x{overlay_height}")
            
            self.overlay_ = TransparentOverlay(
                width=overlay_width,
                height=overlay_height,
                window_name="WOT_AI Navigation Overlay",
                pos_x=0,  # 左上角
                pos_y=0,  # 左上角
                fps=self.config_.get('overlay_fps', 30),
                alpha=self.config_.get('overlay_alpha', 180)
            )            

            # self.overlay_ = TransparentOverlay(
            #     width=overlay_width,
            #     height=overlay_height,
            #     window_name="WOT_AI Navigation Overlay",
            #     pos_x=minimap_region['x'],  # 与小地图区域左上角对齐
            #     pos_y=minimap_region['y'],  # 与小地图区域左上角对齐
            #     fps=self.config_.get('overlay_fps', 30),
            #     alpha=self.config_.get('overlay_alpha', 180)
            # )
            
            logger.info("透明覆盖层初始化成功（位于左上角）")
            return True
            
        except Exception as e:
            logger.error(f"透明覆盖层初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ExtractMinimap(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从小地图区域提取小地图图像
        
        Args:
            frame: 屏幕帧
        
        Returns:
            小地图图像，失败返回 None
        """
        if self.minimap_region_ is None:
            return None
        
        x = self.minimap_region_['x']
        y = self.minimap_region_['y']
        w = self.minimap_region_['width']
        h = self.minimap_region_['height']
        
        # 检查边界
        frame_h, frame_w = frame.shape[:2]
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            logger.warning(f"小地图区域超出屏幕范围: ({x}, {y}, {w}, {h}) vs ({frame_w}, {frame_h})")
            return None
        
        minimap = frame[y:y+h, x:x+w]
        return minimap
    
    def _LoadMaskAndAlign(self, mask_path: Path, minimap_frame: np.ndarray, corners_xy: np.ndarray, 
                          inflation_size: int = 10) -> np.ndarray:
        """
        加载掩码并做透视变换对齐到小地图
        
        Args:
            mask_path: 掩码文件路径
            minimap_frame: 小地图图像（BGR）
            corners_xy: 小地图四角坐标，形状为(4,2)，顺序为 [左上, 右上, 右下, 左下]
            inflation_size: 膨胀操作半径，用于扩大障碍物，让路径远离障碍
        
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
        
        # 对障碍物进行膨胀操作，扩大障碍物，让路径远离障碍
        # 注意：膨胀在后续的build_inflated_and_cost_map中会再次进行，这里先不做膨胀
        # 保留原始掩码，让膨胀在代价图构建时统一处理
        
        return mask01
    
    def _MaskToGrid(self, mask_01: np.ndarray, grid_size: tuple) -> np.ndarray:
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
    
    def _WorldToGrid(self, world_pos: tuple, minimap_size: tuple, grid_size: tuple) -> tuple:
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
    
    def LoadMask(self, minimap_size: tuple) -> bool:
        """
        加载掩码（在初始化时调用一次）
        
        Args:
            minimap_size: 小地图尺寸 (width, height)
        
        Returns:
            是否加载成功
        """
        try:
            mask_path = self.config_.get('mask_path')
            grid_size = self.config_.get('grid_size', (256, 256))
            
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
                
                inflation_size = self.config_.get('inflation_radius_px', 20)
                aligned_mask = self._LoadMaskAndAlign(Path(mask_path), temp_minimap, corners_xy, inflation_size)
                
                # 转换为栅格尺寸
                grid = self._MaskToGrid(aligned_mask, grid_size)
                logger.info(f"从掩码构建栅格地图，尺寸: {grid.shape}")
                
                # 构建膨胀障碍图和代价图
                cost_alpha = self.config_.get('cost_alpha', 20.0)
                inflated_obstacle, cost_map = build_inflated_and_cost_map(
                    grid,
                    inflate_radius_px=self.config_.get('inflation_radius_px', 20),
                    alpha=cost_alpha
                )
                
                # 保存对齐后的掩码、栅格地图、代价图和膨胀障碍图
                self.current_aligned_mask_ = aligned_mask
                self.current_grid_ = grid
                self.current_cost_map_ = cost_map
                self.current_inflated_obstacle_ = inflated_obstacle
                logger.info(f"障碍膨胀和代价图构建完成: 膨胀半径={inflation_size}px, cost_alpha={cost_alpha}")
                return True
            else:
                # 如果没有掩码，创建一个空的栅格地图（全部可通行）
                logger.warning("未提供掩码，使用全可通行栅格地图")
                grid = np.zeros((grid_size[1], grid_size[0]), dtype=np.uint8)
                # 即使没有掩码，也构建代价图（全可通行区域）
                cost_alpha = self.config_.get('cost_alpha', 10.0)
                inflated_obstacle, cost_map = build_inflated_and_cost_map(
                    grid,
                    inflate_radius_px=self.config_.get('inflation_radius_px', 10),
                    alpha=cost_alpha
                )
                self.current_aligned_mask_ = None
                self.current_grid_ = grid
                self.current_cost_map_ = cost_map
                self.current_inflated_obstacle_ = inflated_obstacle
                return True
                
        except Exception as e:
            logger.error(f"掩码加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def PlanPath(self, minimap_size: tuple, detections: MinimapDetectionResult) -> Optional[list]:
        """
        规划路径
        
        Args:
            minimap_size: 小地图尺寸 (width, height)
            detections: 检测结果
        
        Returns:
            路径坐标列表（栅格坐标），失败返回 None
        """
        if detections.self_pos is None or detections.enemy_flag_pos is None:
            logger.warning("检测结果不完整，无法规划路径")
            return None
        
        if self.current_grid_ is None:
            logger.error("栅格地图未初始化，请先调用 LoadMask")
            return None
        
        try:
            grid_size = self.config_.get('grid_size', (256, 256))
            
            # 将检测结果的位置转换为栅格坐标
            start = self._WorldToGrid(detections.self_pos, minimap_size, grid_size)
            goal = self._WorldToGrid(detections.enemy_flag_pos, minimap_size, grid_size)
            
            logger.info(f"起点（栅格坐标）: {start}, 终点（栅格坐标）: {goal}")
            
            # 使用带代价图的A*算法规划路径
            if self.current_cost_map_ is not None:
                # 使用新的cost_map A*
                path = astar_with_cost(self.current_cost_map_, start, goal)
                if not path:
                    logger.warning("cost_map A*规划失败，尝试使用传统A*")
                    path = self.planner_.Plan(self.current_grid_, start, goal)
                else:
                    # LOS平滑
                    if self.config_.get('enable_los_smoothing', True) and self.current_inflated_obstacle_ is not None:
                        path = smooth_path_los(path, self.current_inflated_obstacle_)
            else:
                # 回退到传统A*
                logger.warning("代价图未初始化，使用传统A*算法")
                path = self.planner_.Plan(self.current_grid_, start, goal)
            
            if path:
                logger.info(f"路径规划成功，路径长度: {len(path)}")
            else:
                logger.warning("路径规划失败")
            
            return path
            
        except Exception as e:
            logger.error(f"路径规划失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def UpdateOverlay(self, path: list, detections: MinimapDetectionResult):
        """
        更新透明覆盖层显示
        
        Args:
            path: 路径坐标列表（栅格坐标）
            detections: 检测结果
        """
        if self.overlay_ is None:
            return
        
        import time
        start_time = time.time()
        
        # 准备检测结果字典
        detections_dict = {
            'self_pos': detections.self_pos,
            'flag_pos': detections.enemy_flag_pos,
            'angle': detections.self_angle
        }
        
        # 从minimap_region_获取尺寸信息
        if self.minimap_region_ is None:
            return
        
        minimap_size = (self.minimap_region_['width'], self.minimap_region_['height'])
        grid_size = self.config_.get('grid_size', (256, 256))
        
        # 更新覆盖层（传递掩码用于显示，不需要minimap图像）
        self.overlay_.DrawPath(
            minimap=None,
            path=path,
            detections=detections_dict,
            minimap_size=minimap_size,
            grid_size=grid_size,
            mask=self.current_aligned_mask_
        )
        
        elapsed = time.time() - start_time
        if elapsed > 0.01:  # 只记录超过10ms的更新
            logger.debug(f"UpdateOverlay耗时: {elapsed*1000:.2f}ms")
    
    def _DetectionThread_(self):
        """检测线程：负责屏幕捕获、小地图提取、YOLO检测"""
        logger.info("检测线程启动")
        
        detection_fps = 15  # 10FPS（小地图检测不需要30FPS）
        detection_interval = 1.0 / detection_fps  # 100ms
        
        # FPS计算
        fps_window_size = 10  # 计算FPS的窗口大小（10帧）
        frame_times = []  # 存储最近N帧的时间
        
        while self.running_:
            try:
                start_time = time.time()
                
                # 1. 捕获屏幕
                capture_start = time.time()
                minimap = self.capture_service_.Capture(self.minimap_region_)
                capture_elapsed = time.time() - capture_start
                if minimap is None:
                    time.sleep(detection_interval)
                    continue
                
                # 2. YOLO检测
                detect_start = time.time()
                detections = self.minimap_detector_.Detect(minimap, debug=False)
                detect_elapsed = time.time() - detect_start
                
                # 性能监控：分别记录capture和detect耗时
                if capture_elapsed > 0.05:  # 如果屏幕捕获超过50ms，记录警告
                    logger.warning(f"屏幕捕获耗时: {capture_elapsed*1000:.2f}ms")
                if detect_elapsed > 0.1:  # 如果YOLO检测超过100ms，记录警告
                    logger.warning(f"YOLO检测耗时: {detect_elapsed*1000:.2f}ms")
                # logger.info(f"检测结果: {detections}")
                # 放宽条件：允许只有self_pos的结果进入队列（UI只需要显示自己位置）
                if detections.self_pos is None:
                    time.sleep(detection_interval)
                    continue
                
                # 非阻塞放入队列（队列满时丢弃旧数据）
                try:
                    self.detection_queue_.put_nowait((detections))
                except queue.Full:
                    # 队列满，丢弃最旧的数据，放入新数据
                    try:
                        self.detection_queue_.get_nowait()
                        self.detection_queue_.put_nowait((detections))
                    except queue.Empty:
                        pass
                
                # 计算FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                if len(frame_times) > fps_window_size:
                    frame_times.pop(0)
                
                # 计算平均FPS并更新overlay
                if len(frame_times) > 0 and self.overlay_:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    # 更新检测线程FPS（保持UI FPS不变）
                    if hasattr(self.overlay_, 'current_ui_fps_'):
                        self.overlay_.UpdateFps(self.overlay_.current_ui_fps_, current_fps)
                
                # 控制循环频率
                if elapsed > detection_interval * 2:  # 如果检测耗时超过目标间隔的2倍，记录警告
                    logger.warning(f"检测耗时过长: {elapsed*1000:.2f}ms (目标: {detection_interval*1000:.2f}ms)")
                sleep_time = max(0, detection_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"检测线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(detection_interval)
        
        logger.info("检测线程退出")
    
    def _ControlThread_(self):
        """控制线程：执行路径规划和移动控制（30FPS，完全无锁）"""
        logger.info("控制线程启动")
        
        # 获取配置参数
        stuck_threshold = self.config_.get('stuck_threshold', 5.0)
        stuck_frames_threshold = self.config_.get('stuck_frames_threshold', 10)
        target_point_offset = self.config_.get('target_point_offset', 5)
        control_fps = 30  # 30FPS
        control_interval = 1.0 / control_fps  # 约33ms
        
        # 本地状态维护（完全无锁）
        local_last_pos_ = None
        local_stuck_frames_ = 0
        local_current_path_ = []
        local_current_target_idx_ = 0
        
        while self.running_:
            try:
                start_time = time.time()
                
                # 清空队列，只取最新的检测结果
                detections = None
                while not self.detection_queue_.empty():
                    try:
                        detections = self.detection_queue_.get_nowait()
                    except queue.Empty:
                        break
                
                if detections is None:
                    time.sleep(control_interval)
                    continue
                
                current_pos = detections.self_pos
                if current_pos is None:
                    time.sleep(control_interval)
                    continue
                
                # 从检测结果获取角度（度）并转换为弧度
                # extract_arrow_orientation 返回的角度：0° 表示向右，逆时针为正
                current_heading_rad = None
                if detections.self_angle is not None:
                    # 将角度从度转换为弧度
                    # 注意：extract_arrow_orientation 返回的是 0-360 度的角度
                    # 需要转换为标准数学角度（0°向右，逆时针为正，范围 -180 到 180）
                    angle_deg = detections.self_angle
                    # 转换为弧度：0° = 0 rad, 90° = π/2 rad, 180° = π rad, 270° = -π/2 rad
                    current_heading_rad = math.radians(angle_deg)
                    logger.info(f"当前角度: {angle_deg:.2f}°")
                
                # 检测卡顿（基于位移）
                is_stuck = False
                if local_last_pos_ is not None:
                    # 只检测当前位置周围小范围的位移
                    distance = math.sqrt(
                        (current_pos[0] - local_last_pos_[0]) ** 2 + 
                        (current_pos[1] - local_last_pos_[1]) ** 2
                    )
                    
                    if distance < stuck_threshold:
                        local_stuck_frames_ += 1
                        if local_stuck_frames_ >= stuck_frames_threshold:
                            is_stuck = True
                            logger.warning(f"检测到卡顿，连续{local_stuck_frames_}帧位移小于{stuck_threshold}像素")
                    else:
                        local_stuck_frames_ = 0
                
                local_last_pos_ = current_pos
                
                # 判断是否需要重新规划
                need_replan = (
                    len(local_current_path_) == 0 or
                    is_stuck or
                    local_current_target_idx_ >= len(local_current_path_) - 1
                )
                
                if need_replan:
                    logger.info("触发重新规划")
                    # 停止移动
                    self.nav_executor_.StopMoving()
                    
                    # 获取小地图尺寸
                    minimap_h, minimap_w = self.minimap_region_['height'], self.minimap_region_['width']
                    minimap_size = (minimap_w, minimap_h)
                    
                    # 规划路径（只需要尺寸，不需要图像）
                    path = self.PlanPath(minimap_size, detections)
                    if path is None or len(path) == 0:
                        logger.warning("路径规划失败或路径为空")
                        time.sleep(control_interval)
                        continue
                    
                    local_current_path_ = path
                    local_current_target_idx_ = 0
                    local_stuck_frames_ = 0
                    logger.info(f"路径规划成功，路径长度: {len(path)}")
                    
                    # 将路径放入队列（非阻塞，队列满时替换）
                    try:
                        self.path_queue_.put_nowait(path)
                    except queue.Full:
                        # 队列满，替换
                        try:
                            self.path_queue_.get_nowait()
                            self.path_queue_.put_nowait(path)
                        except queue.Empty:
                            pass
                
                # 执行持续移动控制
                if len(local_current_path_) > 1:
                    # 将栅格坐标转换回小地图坐标的缩放比例
                    minimap_h, minimap_w = self.minimap_region_['height'], self.minimap_region_['width']
                    grid_size = self.config_.get('grid_size', (256, 256))
                    scale_x = minimap_w / grid_size[0]
                    scale_y = minimap_h / grid_size[1]
                    
                    # 根据target_point_offset选择路径上的远端目标点
                    target_idx = min(
                        local_current_target_idx_ + target_point_offset,
                        len(local_current_path_) - 1
                    )
                    
                    if target_idx < len(local_current_path_):
                        target_point = local_current_path_[target_idx]
                        target_world_x = target_point[0] * scale_x
                        target_world_y = target_point[1] * scale_y
                        target_world = (target_world_x, target_world_y)
                        
                        # 计算到目标点的距离
                        distance_to_target = math.sqrt(
                            (target_world[0] - current_pos[0]) ** 2 + 
                            (target_world[1] - current_pos[1]) ** 2
                        )
                        
                        # 如果接近目标点，更新索引
                        arrival_threshold = self.config_.get('arrival_threshold', 20.0)
                        if distance_to_target < arrival_threshold:
                            local_current_target_idx_ = min(target_idx + 1, len(local_current_path_) - 1)
                        else:
                            # 持续转向目标点（使用A/D键）
                            # 使用检测到的角度，如果为None则使用默认值0.0
                            heading = current_heading_rad if current_heading_rad is not None else 0.0
                            self.nav_executor_.RotateToward(target_world, current_pos, heading)
                            # 确保正在持续前进
                            self.nav_executor_.EnsureMovingForward()
                    else:
                        # 路径已耗尽，停止移动
                        self.nav_executor_.StopMoving()
                        logger.info("已到达路径终点")
                
                # 控制循环频率（30FPS）
                elapsed = time.time() - start_time
                sleep_time = max(0, control_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"控制线程错误: {e}")
                import traceback
                traceback.print_exc()
                # 确保异常时释放按键
                if self.nav_executor_:
                    self.nav_executor_.StopMoving()
                time.sleep(control_interval)
        
        logger.info("控制线程退出")
    
    def _UIThread_(self):
        """UI更新线程：更新透明覆盖层显示（30FPS，完全无锁）"""
        logger.info("UI更新线程启动")
        
        ui_fps = 30  # 30FPS
        ui_interval = 1.0 / ui_fps  # 约33ms
        
        # FPS计算
        fps_window_size = 30  # 计算FPS的窗口大小（30帧）
        frame_times = []  # 存储最近N帧的时间
        
        while self.running_:
            try:
                start_time = time.time()
                
                # 清空检测队列，只取最新的检测结果
                detections = None
                while not self.detection_queue_.empty():
                    try:
                        detections = self.detection_queue_.get_nowait()
                    except queue.Empty:
                        break
                
                # 从路径队列获取最新路径（非阻塞，队列为空时使用缓存）
                if not hasattr(self, "_ui_cached_path_"):
                    self._ui_cached_path_ = None
                try:
                    current_path = self.path_queue_.get_nowait()
                    self._ui_cached_path_ = current_path
                except queue.Empty:
                    current_path = self._ui_cached_path_
                
                # 初始化检测结果缓存
                if not hasattr(self, "_ui_cached_detections_"):
                    self._ui_cached_detections_ = {}
                
                # 准备参数（无论是否有检测结果，都需要更新路径）
                minimap_size = None
                if self.minimap_region_:
                    minimap_size = (self.minimap_region_['width'], self.minimap_region_['height'])
                
                grid_size = None
                if self.current_grid_ is not None:
                    grid_size = (self.current_grid_.shape[1], self.current_grid_.shape[0])  # (width, height)
                
                # 更新检测结果缓存：只更新非None的字段
                if detections is not None:
                    if hasattr(detections, 'self_pos') and detections.self_pos is not None:
                        self._ui_cached_detections_['self_pos'] = detections.self_pos
                    if hasattr(detections, 'self_angle') and detections.self_angle is not None:
                        self._ui_cached_detections_['self_angle'] = detections.self_angle
                    if hasattr(detections, 'enemy_flag_pos') and detections.enemy_flag_pos is not None:
                        self._ui_cached_detections_['flag_pos'] = detections.enemy_flag_pos
                
                # 使用缓存值（即使新检测结果为None）
                detections_dict = self._ui_cached_detections_.copy()
                
                # 更新UI显示（路径、检测结果、掩码背景）
                # 即使没有检测结果，也要更新路径（使用空的detections_dict）
                if self.overlay_:
                    self.overlay_.DrawPath(
                        minimap=None,  # 不需要图像，只需要尺寸
                        path=current_path,
                        detections=detections_dict,
                        minimap_size=minimap_size,
                        grid_size=grid_size,
                        mask=self.current_aligned_mask_
                    )
                
                # 计算FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                if len(frame_times) > fps_window_size:
                    frame_times.pop(0)
                
                # 计算平均FPS
                if len(frame_times) > 0:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    
                    # 更新overlay的FPS显示
                    if self.overlay_:
                        # 检测线程FPS（从检测队列更新频率估算）
                        detection_fps = 0.0
                        if hasattr(self, 'detection_queue_'):
                            # 简单估算：如果队列有数据，说明检测线程在运行
                            # 实际检测FPS需要从检测线程传递，这里先设为0
                            pass
                        self.overlay_.UpdateFps(current_fps, detection_fps)
                
                # 控制循环频率（30FPS）
                if elapsed > ui_interval * 1.5:  # 如果实际耗时超过目标间隔的1.5倍，记录警告
                    logger.warning(f"UI更新耗时过长: {elapsed*1000:.2f}ms (目标: {ui_interval*1000:.2f}ms)")
                sleep_time = max(0, ui_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"UI更新线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(ui_interval)
        
        logger.info("UI更新线程退出")
    
    def Run(self):
        """主循环：启动多线程架构"""
        logger.info("启动导航主循环（多线程模式）...")
        self.running_ = True
        
        # 首次检测小地图位置
        logger.info("首次检测小地图位置...")
        frame = self.capture_service_.Capture()
        if frame is None:
            logger.error("无法捕获屏幕")
            return
        
        self.minimap_region_ = self.DetectMinimapRegion(frame)
        if self.minimap_region_ is None:
            logger.error("无法检测到小地图位置，退出")
            return
        
        # 加载掩码（在初始化时调用一次）
        minimap_size = (self.minimap_region_['width'], self.minimap_region_['height'])
        if not self.LoadMask(minimap_size):
            logger.error("掩码加载失败，退出")
            return
        
        # 初始化透明覆盖层（必须与小地图区域完全重合）
        if not self.InitializeOverlay(self.minimap_region_):
            logger.error("透明覆盖层初始化失败，退出")
            return
        
        # 初始化状态
        self.current_target_idx_ = 0
        self.stuck_frames_ = 0
        
        disable_control = False

        # 启动线程
        logger.info("启动检测线程、控制线程和UI更新线程...")
        self.detection_thread_ = threading.Thread(target=self._DetectionThread_, daemon=True)
        self.ui_thread_ = threading.Thread(target=self._UIThread_, daemon=True)
        
        self.detection_thread_.start()
        if disable_control:
            self.control_thread_ = None
        else:
            self.control_thread_ = threading.Thread(target=self._ControlThread_, daemon=True)
            self.control_thread_.start()

        self.ui_thread_.start()
        
        logger.info("所有线程已启动，等待线程运行...")
        
        # 主线程等待所有线程完成
        try:
            while self.running_:
                time.sleep(5)
                # 检查线程是否还在运行
                if not self.detection_thread_.is_alive():
                    logger.warning("检测线程已退出")
                if self.control_thread_ and not self.control_thread_.is_alive():
                    logger.warning("控制线程已退出")
                if not self.ui_thread_.is_alive():
                    logger.warning("UI更新线程已退出")
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号")
        finally:
            logger.info("正在停止所有线程...")
            self.running_ = False
            
            # 等待线程结束
            if self.detection_thread_ and self.detection_thread_.is_alive():
                self.detection_thread_.join(timeout=1.0)
            if self.control_thread_ and self.control_thread_.is_alive():
                self.control_thread_.join(timeout=1.0)
            if self.ui_thread_ and self.ui_thread_.is_alive():
                self.ui_thread_.join(timeout=1.0)
        
        logger.info("导航主循环结束")
    
    def Stop(self):
        """停止运行"""
        logger.info("正在停止...")
        self.running_ = False
        
        # 等待线程结束
        if self.detection_thread_ and self.detection_thread_.is_alive():
            self.detection_thread_.join(timeout=1.0)
        if self.control_thread_ and self.control_thread_.is_alive():
            self.control_thread_.join(timeout=1.0)
        if self.ui_thread_ and self.ui_thread_.is_alive():
            self.ui_thread_.join(timeout=1.0)
        
        # 确保释放所有按键
        if self.nav_executor_:
            self.nav_executor_.StopMoving()
            self.nav_executor_.Stop()
        
        if self.overlay_:
            self.overlay_.Close()
        
        logger.info("已停止")


def main():
    """主函数"""
    # 配置参数
    script_dir = Path(__file__).resolve().parent.parent.parent
    config = {
        # 模型路径
        'model_path': str(script_dir / "tests" / "best_s_seg.pt"),
        # 小地图模板路径
        'template_path': str(script_dir / "tests" / "minimap_border.png"),
        # 小地图尺寸（用于初始化TransparentOverlay）
        'minimap_size': (640, 640),
        # 掩码路径（可选）
        'mask_path': str(script_dir / "tests" / "胜利之门_mask.png"),
        # 栅格尺寸
        'grid_size': (256, 256),
        # 障碍膨胀参数
        'inflation_radius_px': 10,  # 障碍膨胀半径（像素）
        'cost_alpha': 10.0,  # 代价图权重
        'enable_los_smoothing': True,  # 启用LOS平滑
        # YOLO检测参数
        'conf_threshold': 0.25,
        'iou_threshold': 0.75,
        # 路径规划参数
        'enable_smoothing': True,
        'smooth_weight': 0.3,
        # 控制参数
        'move_speed': 1.0,
        'rotation_smooth': 0.3,
        # 持续移动控制参数
        'target_point_offset': 5,  # 目标点偏移量（选择路径上第几个点作为目标）
        'arrival_threshold': 20.0,  # 到达阈值（像素）
        # 解卡参数
        'stuck_threshold': 5.0,  # 卡顿检测阈值（像素）
        'stuck_frames_threshold': 10,  # 连续卡顿帧数阈值
        'stuck_detection_radius': 10.0,  # 卡顿检测范围（像素）
        'heading_change_threshold': 0.1,  # 朝向变化阈值（弧度，约5.7度）
        # UI参数
        'overlay_fps': 30,
        'overlay_alpha': 180,
        # 循环参数
        'loop_interval': 0.05  # 循环间隔（秒）
    }
    
    # 创建主程序实例
    nav_main = NavigationMain(config)
    
    # 初始化
    if not nav_main.Initialize():
        logger.error("初始化失败，退出")
        return 1
    
    # 运行主循环
    try:
        nav_main.Run()
    except Exception as e:
        logger.error(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        nav_main.Stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

