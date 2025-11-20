#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NavigationRuntime v1
两线程 + DataHub（检测线程 + 控制线程）
ThreadManager 的替代方案（不带 UI）
"""

import math
import time
import threading
from typing import Optional, Tuple

from loguru import logger

from src.navigation.service.data_hub import DataHub
from src.navigation.service.capture_service import CaptureService
from src.navigation.service.minimap_service import MinimapService
from src.navigation.service.path_planning_service import PathPlanningService
from src.navigation.core.navigation_executor import NavigationExecutor
from src.navigation.core.path_follower import PathFollower
from src.navigation.config.models import NavigationConfig

from src.navigation.nav_runtime.stuck_detector import StuckDetector
from src.navigation.nav_runtime.path_planner_wrapper import PathPlannerWrapper
from src.navigation.nav_runtime.movement_controller import MovementController
from src.navigation.nav_runtime.path_follower_wrapper import PathFollowerWrapper
from src.vision.minimap_detector import MinimapDetector


class NavigationRuntime:
    """
    导航运行时：
    - detection_thread: 捕获 + YOLO 检测（尽可能快）
    - control_thread: 路径规划 + 路径跟随 + 转向 + 前进（固定 tick）
    - DataHub: 共享最新检测结果
    """

    def __init__(self):
        self.cfg = NavigationConfig()
        self.stuck_detector = StuckDetector()
        self.move = MovementController(NavigationExecutor())
        self.path_follower = PathFollower()
        self.path_follower_wrapper = PathFollowerWrapper(
            follower=self.path_follower,
            deviation_tolerance=self.cfg.control.path_deviation_tolerance,
            target_point_offset=self.cfg.control.target_point_offset,
            goal_arrival_threshold=self.cfg.control.goal_arrival_threshold,
        )

        self.capture = CaptureService()
        self.detector = MinimapDetector()
        self.minimap_service = MinimapService()
        self.planner = PathPlanningService()

        self.data_hub = DataHub()

        # 线程
        self._running = False
        self._det_thread: Optional[threading.Thread] = None
        self._ctrl_thread: Optional[threading.Thread] = None

        # 路径跟随状态（仅控制线程内部访问）
        self.current_path_grid = []
        self.current_path_world = []
        self.current_target_idx = 0

    # ============================================================
    # 外部接口
    # ============================================================
    def start(self):
        if self._running:
            logger.warning("NavigationRuntime 已在运行")
            return False

        self.minimap_service.minimap_region = self.minimap_service.detect_region(self.capture.grab())
        if not self.minimap_service.minimap_region:
            logger.error("无法检测到小地图位置")
            return False
        logger.info(f"检测到小地图区域: {self.minimap_service.minimap_region}")

        self._running = True

        self._det_thread = threading.Thread(target=self._det_loop, daemon=True)
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)

        self._det_thread.start()
        self._ctrl_thread.start()

        logger.info("NavigationRuntime 已启动（检测 + 控制）")
        return True

    def stop(self):
        if not self._running:
            return

        logger.info("正在停止 NavigationRuntime ...")
        self._running = False

        if self._det_thread and self._det_thread.is_alive():
            self._det_thread.join(timeout=1.0)

        if self._ctrl_thread and self._ctrl_thread.is_alive():
            self._ctrl_thread.join(timeout=1.0)

        # 停止所有按键
        try:
            self.move.stop()
        except Exception:
            pass

        logger.info("NavigationRuntime 已停止")

    def is_running(self):
        return self._running

    # ============================================================
    # 检测线程：尽可能快地产生最新 detection
    # ============================================================
    def _det_loop(self):
        logger.info("检测线程启动")

        detect_fps = getattr(self.cfg.performance, "detect_fps", 30)
        min_interval = 1.0 / detect_fps

        x, y, w, h = self.minimap_service.minimap_region["x"], self.minimap_service.minimap_region["y"], self.minimap_service.minimap_region["width"], self.minimap_service.minimap_region["height"]
        
        while self._running:
            t0 = time.perf_counter()
            try:
                frame = self.capture.grab_region(x, y, w, h)
                if frame is None:
                    time.sleep(0.001)
                    continue

                # 2. YOLO 检测
                det = self.detector.Detect(frame)
                if det is None or det.self_pos is None:
                    continue

                # 3. 写入 DataHub
                self.data_hub.set_latest_detection(det)

            except Exception as e:
                logger.error(f"检测线程错误: {e}")
                import traceback
                traceback.print_exc()

            # 限制最大 FPS
            dt = time.perf_counter() - t0
            if dt < min_interval:
                time.sleep(min_interval - dt)

        logger.info("检测线程退出")

        
    def _ctrl_loop(self):
        logger.info("控制线程启动")
    
        # 控制线程 FPS
        ctrl_fps = getattr(self.cfg.performance, "control_fps", 20)
        interval = 1.0 / ctrl_fps
    
        minimap = self.minimap_service.minimap_region
        if not minimap:
            logger.error("minimap_region 未配置，控制线程退出")
            return
    
        # grid -> minimap 缩放
        grid_w, grid_h = self.cfg.grid.size
        mmap_w = minimap["width"]
        mmap_h = minimap["height"]
        sx = mmap_w / grid_w
        sy = mmap_h / grid_h
    
        # 包装路径规划器
        self.path_planner = PathPlannerWrapper(
            planner=self.planner,
            minimap_size=(mmap_w, mmap_h),
            scale_xy=(sx, sy),
        )
    
        while self._running:
            t0 = time.perf_counter()
    
            # ============================================================
            # 1) 读取最新检测结果
            # ============================================================
            det = self.data_hub.get_latest_detection(max_age=0.5)
            if det is None or det.self_pos is None:
                self.move.stop()
                time.sleep(interval)
                continue
            
            pos = det.self_pos
            heading = math.radians(det.self_angle) if det.self_angle else 0.0
    
            # ============================================================
            # 2) 卡顿检测 + 重规划判断
            # ============================================================
            is_stuck = self.stuck_detector.update(pos)
    
            need_replan = (
                not self.current_path_world
                or is_stuck
                or self.current_target_idx >= len(self.current_path_world) - 1
            )
    
            # ============================================================
            # 3) 路径规划（若需要）
            # ============================================================
            if need_replan:
                self.move.stop()
                grid_path, world_path = self.path_planner.plan(det)
    
                if not world_path:
                    # 规划失败
                    self.move.stop()
                    time.sleep(interval)
                    continue
                
                # 更新内部路径状态
                self.current_path_grid = grid_path
                self.current_path_world = world_path
                self.current_target_idx = 0
    
                # 同步到 DataHub
                try:
                    self.data_hub.set_current_path(
                        grid_path=grid_path,
                        world_path=world_path,
                        target_idx=0,
                    )
                except Exception:
                    pass
                
                # 规划后重置卡顿检测
                self.stuck_detector.reset()
    
                logger.info(f"新路径规划节点数：{len(world_path)}")
    
            # ============================================================
            # 4) 路径跟随：计算下一控制目标
            # ============================================================
            (
                target_world,
                dev,
                dist_goal,
                goal_reached,
                new_idx,
                used_target_idx,
            ) = self.path_follower_wrapper.follow(
                current_pos=pos,
                path_world=self.current_path_world,
                current_target_idx=self.current_target_idx,
            )
    
            # 更新当前索引
            self.current_target_idx = new_idx
    
            # ============================================================
            # 5) 更新 DataHub 状态
            # ============================================================
            try:
                self.data_hub.set_nav_status(
                    is_stuck=is_stuck,
                    stuck_frames=self.stuck_detector.stuck_frames,
                    path_deviation=dev,
                    distance_to_goal=dist_goal,
                    goal_reached=goal_reached,
                )
    
                self.data_hub.set_current_path(
                    grid_path=self.current_path_grid,
                    world_path=self.current_path_world,
                    target_idx=self.current_target_idx,
                )
            except Exception:
                pass
            
            # ============================================================
            # 6) 终点判断
            # ============================================================
            if goal_reached or target_world is None:
                self.move.stop()
    
                # 清空路径让下次必然触发重规划
                self.current_path_grid = []
                self.current_path_world = []
                self.current_target_idx = 0
    
                time.sleep(interval)
                continue
            
            # ============================================================
            # 7) 控制执行（MovementController）
            # ============================================================
            self.move.goto(
                target_pos=target_world,
                current_pos=pos,
                heading=heading
            )
    
            # 控制 tick
            dt = time.perf_counter() - t0
            if dt < interval:
                time.sleep(interval - dt)
    
        logger.info("控制线程退出")

if __name__ == "__main__":
    nav_runtime = NavigationRuntime()
    nav_runtime.start()
    logger.info("NavigationRuntime 已停止")