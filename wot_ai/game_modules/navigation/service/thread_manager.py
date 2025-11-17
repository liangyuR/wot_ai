#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线程管理器模块

封装三个工作线程的管理逻辑，提供线程启动、停止、状态查询接口。
"""

import time
import math
import threading
import queue
from typing import Optional
import numpy as np
from loguru import logger

from wot_ai.game_modules.navigation.service.capture_service import CaptureService
from wot_ai.game_modules.vision.detection.minimap_detector import MinimapDetectionResult
from wot_ai.game_modules.navigation.service.minimap_service import MinimapService
from wot_ai.game_modules.navigation.service.path_planning_service import PathPlanningService
from wot_ai.game_modules.navigation.core.navigation_executor import NavigationExecutor
from wot_ai.game_modules.navigation.core.path_follower import PathFollower
from wot_ai.game_modules.navigation.core.coordinate_utils import grid_to_world
from wot_ai.game_modules.navigation.config.models import NavigationConfig


class ThreadManager:
    """线程管理器"""
    
    def __init__(
        self,
        capture_service: CaptureService,
        minimap_detector,
        minimap_service: MinimapService,
        path_planning_service: PathPlanningService,
        nav_executor: NavigationExecutor,
        queues: dict,
        mask_data,
        config: NavigationConfig
    ):
        """
        初始化线程管理器
        
        Args:
            capture_service: 屏幕捕获服务
            minimap_detector: 小地图检测器
            minimap_service: 小地图服务
            path_planning_service: 路径规划服务
            nav_executor: 导航执行器
            queues: 队列字典 {'detection': Queue, 'path': Queue}
            mask_data: 掩码数据对象
            config: 配置字典
        """
        self.capture_service_ = capture_service
        self.minimap_detector_ = minimap_detector
        self.minimap_service_ = minimap_service
        self.path_planning_service_ = path_planning_service
        self.nav_executor_ = nav_executor
        self.detection_queue_ = queues['detection']
        self.path_queue_ = queues['path']
        self.mask_data_ = mask_data
        self.config_ = config
        
        self.running_ = False
        self.detection_thread_: Optional[threading.Thread] = None
        self.control_thread_: Optional[threading.Thread] = None
        self.ui_thread_: Optional[threading.Thread] = None
        
        self.path_follower_ = PathFollower()
        
        # UI线程缓存
        self._ui_cached_path_ = None
        self._ui_cached_detections_ = {}
    
    def start_all(self) -> bool:
        """
        启动所有线程
        
        Returns:
            是否成功启动
        """
        if self.running_:
            logger.warning("线程管理器已在运行")
            return False
        
        self.running_ = True
        
        # 启动三个线程
        self.detection_thread_ = threading.Thread(target=self._detection_thread, daemon=True)
        self.control_thread_ = threading.Thread(target=self._control_thread, daemon=True)
        self.ui_thread_ = threading.Thread(target=self._ui_thread, daemon=True)
        
        self.detection_thread_.start()
        self.control_thread_.start()
        self.ui_thread_.start()
        
        logger.info("所有线程已启动")
        return True
    
    def stop_all(self) -> None:
        """停止所有线程"""
        if not self.running_:
            return
        
        logger.info("正在停止所有线程...")
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
        
        logger.info("所有线程已停止")
    
    def is_running(self) -> bool:
        """
        检查是否正在运行
        
        Returns:
            是否正在运行
        """
        return self.running_
    
    def _detection_thread(self) -> None:
        """检测线程：负责屏幕捕获、小地图提取、YOLO检测"""
        logger.info("检测线程启动")
        
        detection_fps = 30
        detection_interval = 1.0 / detection_fps
        
        fps_window_size = 10
        frame_times = []
        
        minimap_region = self.minimap_service_.minimap_region
        
        while self.running_:
            try:
                start_time = time.time()
                
                # 1. 捕获屏幕
                capture_start = time.time()
                minimap = self.capture_service_.Capture(minimap_region)
                capture_elapsed = time.time() - capture_start
                if minimap is None:
                    time.sleep(detection_interval)
                    continue
                
                # 2. YOLO检测
                detect_start = time.time()
                detections = self.minimap_detector_.Detect(minimap, debug=False)
                detect_elapsed = time.time() - detect_start
                
                # 性能监控
                if capture_elapsed > 0.05:
                    logger.warning(f"屏幕捕获耗时: {capture_elapsed*1000:.2f}ms")
                if detect_elapsed > 0.1:
                    logger.warning(f"YOLO检测耗时: {detect_elapsed*1000:.2f}ms")
                
                if detections.self_pos is None:
                    time.sleep(detection_interval)
                    continue
                
                # 非阻塞放入队列
                try:
                    self.detection_queue_.put_nowait(detections)
                except queue.Full:
                    try:
                        self.detection_queue_.get_nowait()
                        self.detection_queue_.put_nowait(detections)
                    except queue.Empty:
                        pass
                
                # 计算FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                if len(frame_times) > fps_window_size:
                    frame_times.pop(0)
                
                # 更新overlay FPS
                if len(frame_times) > 0:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    overlay = self.minimap_service_.overlay
                    if overlay and hasattr(overlay, 'current_ui_fps_'):
                        overlay.UpdateFps(overlay.current_ui_fps_, current_fps)
                
                # 控制循环频率
                if elapsed > detection_interval * 2:
                    logger.warning(f"检测耗时过长: {elapsed*1000:.2f}ms")
                sleep_time = max(0, detection_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"检测线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(detection_interval)
        
        logger.info("检测线程退出")
    
    def _control_thread(self) -> None:
        """控制线程：执行路径规划和移动控制"""
        logger.info("控制线程启动")
        
        stuck_threshold = self.config_.control.stuck_threshold
        stuck_frames_threshold = self.config_.control.stuck_frames_threshold
        control_fps = 30
        control_interval = 1.0 / control_fps
        
        # 本地状态维护
        local_last_pos_ = None
        local_stuck_frames_ = 0
        local_current_path_ = []
        local_current_target_idx_ = 0
        
        minimap_region = self.minimap_service_.minimap_region
        
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
                
                # 获取角度
                current_heading_rad = None
                if detections.self_angle is not None:
                    angle_deg = detections.self_angle
                    current_heading_rad = math.radians(angle_deg)
                    logger.info(f"当前角度: {angle_deg:.2f}°")
                
                # 检测卡顿
                is_stuck = False
                if local_last_pos_ is not None:
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
                    self.nav_executor_.StopMoving()
                    
                    if minimap_region:
                        minimap_h, minimap_w = minimap_region['height'], minimap_region['width']
                        minimap_size = (minimap_w, minimap_h)
                        
                        path = self.path_planning_service_.plan_path(minimap_size, detections)
                        if path is None or len(path) == 0:
                            logger.warning("路径规划失败或路径为空")
                            time.sleep(control_interval)
                            continue
                        
                        local_current_path_ = path
                        local_current_target_idx_ = 0
                        local_stuck_frames_ = 0
                        logger.info(f"路径规划成功，路径长度: {len(path)}")
                        
                        # 将路径放入队列
                        try:
                            self.path_queue_.put_nowait(path)
                        except queue.Full:
                            try:
                                self.path_queue_.get_nowait()
                                self.path_queue_.put_nowait(path)
                            except queue.Empty:
                                pass
                
                # 执行持续移动控制
                if len(local_current_path_) > 1 and minimap_region:
                    minimap_h, minimap_w = minimap_region['height'], minimap_region['width']
                    grid_size = self.config_.grid.size
                    scale_x = minimap_w / grid_size[0]
                    scale_y = minimap_h / grid_size[1]
                    
                    # 将路径转换为世界坐标
                    path_world = []
                    for p in local_current_path_:
                        path_world.append((p[0] * scale_x, p[1] * scale_y))
                    
                    # 找到最近点
                    path_deviation_tolerance = self.config_.control.path_deviation_tolerance
                    nearest_idx, nearest_point, distance_to_path = self.path_follower_.find_nearest_point(
                        current_pos, path_world, max(0, local_current_target_idx_ - 5)
                    )
                    
                    if nearest_idx > local_current_target_idx_:
                        local_current_target_idx_ = nearest_idx
                    
                    if distance_to_path > path_deviation_tolerance * 2:
                        logger.warning(f"路径偏离较大: {distance_to_path:.1f}px")
                    
                    # 选择前瞻目标点
                    target_point_offset = self.config_.control.target_point_offset
                    target_idx = min(
                        local_current_target_idx_ + target_point_offset,
                        len(path_world) - 1
                    )
                    
                    if target_idx < len(path_world):
                        target_world = path_world[target_idx]
                        goal_world = path_world[-1]
                        distance_to_goal = math.sqrt(
                            (goal_world[0] - current_pos[0]) ** 2 + 
                            (goal_world[1] - current_pos[1]) ** 2
                        )
                        
                        goal_arrival_threshold = self.config_.control.goal_arrival_threshold
                        if distance_to_goal < goal_arrival_threshold:
                            self.nav_executor_.StopMoving()
                            logger.info("已到达目标点")
                        else:
                            heading = current_heading_rad if current_heading_rad is not None else 0.0
                            self.nav_executor_.RotateToward(target_world, current_pos, heading)
                            self.nav_executor_.EnsureMovingForward()
                    else:
                        self.nav_executor_.StopMoving()
                        logger.info("已到达路径终点")
                
                # 控制循环频率
                elapsed = time.time() - start_time
                sleep_time = max(0, control_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"控制线程错误: {e}")
                import traceback
                traceback.print_exc()
                if self.nav_executor_:
                    self.nav_executor_.StopMoving()
                time.sleep(control_interval)
        
        logger.info("控制线程退出")
    
    def _ui_thread(self) -> None:
        """UI更新线程：更新透明覆盖层显示"""
        logger.info("UI更新线程启动")
        
        ui_fps = 30
        ui_interval = 1.0 / ui_fps
        
        fps_window_size = 30
        frame_times = []
        
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
                
                # 从路径队列获取最新路径
                try:
                    current_path = self.path_queue_.get_nowait()
                    self._ui_cached_path_ = current_path
                except queue.Empty:
                    current_path = self._ui_cached_path_
                
                # 更新检测结果缓存
                if detections is not None:
                    if hasattr(detections, 'self_pos') and detections.self_pos is not None:
                        self._ui_cached_detections_['self_pos'] = detections.self_pos
                    if hasattr(detections, 'self_angle') and detections.self_angle is not None:
                        self._ui_cached_detections_['self_angle'] = detections.self_angle
                    if hasattr(detections, 'enemy_flag_pos') and detections.enemy_flag_pos is not None:
                        self._ui_cached_detections_['flag_pos'] = detections.enemy_flag_pos
                
                detections_dict = self._ui_cached_detections_.copy()
                
                # 准备参数
                minimap_region = self.minimap_service_.minimap_region
                minimap_size = None
                if minimap_region:
                    minimap_size = (minimap_region['width'], minimap_region['height'])
                
                grid_size = None
                if self.mask_data_ and self.mask_data_.grid is not None:
                    grid_size = (self.mask_data_.grid.shape[1], self.mask_data_.grid.shape[0])
                
                # 更新UI显示
                overlay = self.minimap_service_.overlay
                if overlay:
                    overlay.DrawPath(
                        minimap=None,
                        path=current_path,
                        detections=detections_dict,
                        minimap_size=minimap_size,
                        grid_size=grid_size,
                        mask=self.mask_data_.aligned_mask if self.mask_data_ else None
                    )
                
                # 计算FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                if len(frame_times) > fps_window_size:
                    frame_times.pop(0)
                
                if len(frame_times) > 0:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    
                    if overlay:
                        detection_fps = 0.0
                        overlay.UpdateFps(current_fps, detection_fps)
                
                # 控制循环频率
                if elapsed > ui_interval * 1.5:
                    logger.warning(f"UI更新耗时过长: {elapsed*1000:.2f}ms")
                sleep_time = max(0, ui_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"UI更新线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(ui_interval)
        
        logger.info("UI更新线程退出")

