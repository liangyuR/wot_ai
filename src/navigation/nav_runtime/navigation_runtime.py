#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NavigationRuntime v1 (fixed)
两线程 + DataHub（检测线程 + 控制线程）
ThreadManager 的替代方案（不带 UI）
"""

import math
import random
import time
import threading
from typing import Optional, Tuple

import numpy as np

from loguru import logger
from src.navigation.service.data_hub import DataHub
from src.utils.global_path import GetGlobalConfig
# 屏幕捕获
from src.navigation.service.capture_service import CaptureService
# 运动控制
from src.navigation.nav_runtime.stuck_detector import StuckDetector
from src.navigation.controller.movement_service import MovementService
# 路径规划
from src.navigation.path_planner.path_planning_service import PathPlanningService
from src.navigation.nav_runtime.path_planner_wrapper import PathPlannerWrapper
from src.navigation.nav_runtime.path_follower_wrapper import PathFollowerWrapper
# 小地图识别 & 检测
from src.vision.minimap_anchor_detector import MinimapAnchorDetector
from src.vision.minimap_detector import MinimapDetector
from src.vision.map_name_detector import MapNameDetector
# View
from src.gui.debug_view import DpgNavDebugView


class FrameBuffer:
    """线程安全的最新帧缓冲（单生产者-多消费者）"""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0

    def put(self, frame: np.ndarray) -> None:
        """写入最新帧（由捕获线程调用）"""
        with self._lock:
            self._frame = frame
            self._timestamp = time.perf_counter()

    def get(self) -> Tuple[Optional[np.ndarray], float]:
        """获取最新帧和时间戳（由检测线程调用）"""
        with self._lock:
            return self._frame, self._timestamp


class NavigationRuntime:
    """导航运行时：
    - detection_thread: 捕获 + YOLO 检测（限最高 FPS）
    - control_thread: 路径规划 + 路径跟随 + 转向 + 前进（固定 tick）
    - DataHub: 共享最新检测结果和导航状态
    """

    def __init__(self):
        # 配置
        self.cfg = GetGlobalConfig()

        # 组件
        self.capture = CaptureService(self.cfg.monitor_index)
        if self.cfg.ui.enable:
            self.view = DpgNavDebugView()
        else:
            self.view = None

        # 路径规划
        self.planner_service = PathPlanningService(self.cfg)

        # 数据总线
        self.data_hub = DataHub()

        # 小地图检测
        self.minimap_detector = MinimapDetector()
        self.minimap_anchor_detector = MinimapAnchorDetector()
        self.minimap_name_detector = MapNameDetector()
        self.minimap_region: Optional[dict] = None

        # 控制与跟随
        self.move = MovementService()
        self.path_follower_wrapper = PathFollowerWrapper()
        self.stuck_detector = StuckDetector()

        # 卡顿脱困配置
        self.reverse_duration_s_ = self.cfg.stuck_detection.reverse_duration_s
        self.max_stuck_count_ = self.cfg.stuck_detection.max_stuck_count

        # 线程
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._det_thread: Optional[threading.Thread] = None
        self._ctrl_thread: Optional[threading.Thread] = None
        self._view_thread: Optional[threading.Thread] = None

        # 帧缓冲（捕获线程 -> 检测线程）
        self._frame_buffer: Optional[FrameBuffer] = None

    # ============================================================
    # 外部接口
    # ============================================================
    def start(self, map_name: Optional[str] = None) -> bool:
        """启动导航 Runtime"""
        if self._running:
            logger.warning("NavigationRuntime 已在运行")
            return False

        if not self.minimap_detector.LoadModel():
            logger.error("模型加载失败")
            return False

        # 先检测一次小地图位置，并写入 minimap_region
        # first_frame = self.capture.grab_window_by_name("WorldOfTanks.exe")
        first_frame = self.capture.grab()
        if first_frame is None:
            logger.error("首次抓取屏幕失败，无法检测小地图")
            return False

        self.minimap_region = self.minimap_anchor_detector.DetectRegion(first_frame)
        if not self.minimap_region:
            logger.error("无法检测到小地图位置")
            return False
        logger.info(f"检测到小地图区域: {self.minimap_region}")

        if not map_name:
            from src.utils.key_controller import KeyController
            key_controller = KeyController()
            key_controller.press('b')
            time.sleep(2)
            map_name = self.minimap_name_detector.detect()
            if not map_name:
                logger.error("无法识别地图名称")
            logger.info(f"当前地图名称: {map_name}")
            key_controller.release('b')

        if not self.planner_service.load_map(map_name, (self.minimap_region["width"], self.minimap_region["height"])):
            logger.error("无法加载地图")
            return False
        logger.info(f"地图{map_name}加载成功")

        if self.view is not None:
            self.view.set_grid_mask(self.planner_service.get_grid_mask())

        self._running = True

        # 初始化帧缓冲并启动捕获线程（必须在检测线程之前）
        self._frame_buffer = FrameBuffer()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self._det_thread = threading.Thread(target=self._det_loop, daemon=True)
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)

        self._det_thread.start()
        self._ctrl_thread.start()

        if self.view is not None:
            self._view_thread = threading.Thread(target=self.view.run, daemon=True)
            self._view_thread.start()

        logger.info("NavigationRuntime 已启动（捕获 + 检测 + 控制）")
        return True

    def stop(self) -> None:
        """停止导航 Runtime"""
        if not self._running:
            return

        logger.info("正在停止 NavigationRuntime ...")
        self._running = False

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)

        if self._det_thread and self._det_thread.is_alive():
            self._det_thread.join(timeout=1.0)

        if self._ctrl_thread and self._ctrl_thread.is_alive():
            self._ctrl_thread.join(timeout=1.0)

        if self.view is not None:
            self.view.close()
            if self._view_thread and self._view_thread.is_alive():
                self._view_thread.join(timeout=2.0)

        # 停止所有按键
        try:
            self.move.stop()
        except Exception:
            pass
            
        # 重置内部状态，防止污染下一次运行
        self._reset_internal_state()

        logger.info("NavigationRuntime 已停止")

    def _reset_internal_state(self) -> None:
        """重置内部组件状态"""
        if self.minimap_detector:
            self.minimap_detector.Reset()
        
        if self.stuck_detector:
            self.stuck_detector.reset()
            
        # 如果有其他组件需要重置，可以在这里添加
        # 例如：self.path_follower_wrapper.reset() (如果需要)

    def is_running(self) -> bool:
        return self._running

    # ============================================================
    # 捕获线程：持续抓取小地图区域，写入帧缓冲
    # ============================================================
    def _capture_loop(self) -> None:
        """捕获线程：持续抓取小地图区域"""
        logger.info("捕获线程启动")

        m = self.minimap_region
        x, y, w, h = m["x"], m["y"], m["width"], m["height"]

        while self._running:
            try:
                frame = self.capture.grab_region(x, y, w, h)
                if frame is not None:
                    self._frame_buffer.put(frame)
                # 全速捕获，不做 sleep
            except Exception as e:
                logger.error(f"捕获线程错误: {e}")
                time.sleep(0.01)

        logger.info("捕获线程退出")

    # ============================================================
    # 检测线程：从帧缓冲取帧，YOLO 检测，输出到 DataHub
    # ============================================================
    def _det_loop(self) -> None:
        logger.info("检测线程启动")

        # View 更新降频：每 VIEW_UPDATE_INTERVAL 帧更新一次
        VIEW_UPDATE_INTERVAL = 2
        view_update_counter = 0

        # 记录上一次处理的帧时间戳，避免重复检测同一帧
        last_processed_ts = 0.0

        while self._running:
            try:
                frame, ts = self._frame_buffer.get()
                if frame is None or ts <= last_processed_ts:
                    # 无新帧，短暂等待
                    time.sleep(0.001)
                    continue

                last_processed_ts = ts

                # View 更新（降频）
                view_update_counter += 1
                if self.view is not None and view_update_counter >= VIEW_UPDATE_INTERVAL:
                    self.view.update_minimap_frame(frame)
                    view_update_counter = 0

                det = self.minimap_detector.Detect(frame)
                if det is None or getattr(det, "self_pos", None) is None:
                    continue

                self.data_hub.set_latest_detection(det)

            except Exception as e:
                logger.error(f"检测线程错误: {e}")
                import traceback
                traceback.print_exc()

        logger.info("检测线程退出")

    # ============================================================
    # 控制线程：固定 tick（路径规划 + 路径跟随 + 控制执行）
    # ============================================================
    def _ctrl_loop(self) -> None:
        logger.info("控制线程启动")

        # 控制线程固定 60 FPS
        ctrl_fps = 30
        interval = 1.0 / ctrl_fps

        if not self.minimap_region:
            logger.error("minimap_region 未配置，控制线程退出")
            return

        # grid -> minimap 缩放
        grid_w, grid_h = self.cfg.grid.size
        mmap_w = self.minimap_region["width"]
        mmap_h = self.minimap_region["height"]
        sx = mmap_w / grid_w
        sy = mmap_h / grid_h

        # 包装路径规划器
        self.path_planner = PathPlannerWrapper(
            planner=self.planner_service,
            minimap_size=(mmap_w, mmap_h),
            scale_xy=(sx, sy),
        )

        current_path_grid = []
        current_path_world = []
        current_target_idx = 0

        while self._running:
            t0 = time.perf_counter()

            # 1) 读取最新检测结果
            det = self.data_hub.get_latest_detection(max_age=1.0)
            has_detection = det is not None and getattr(det, "self_pos", None) is not None

            # 更新检测状态（盲走模式逻辑在 MovementService 内部处理）
            self.move.update_detection_status(has_detection)

            if not has_detection:
                # 没有检测结果：由 MovementService 决定是否进入盲走模式
                should_blind_forward = self.move.tick_blind_forward()
                if should_blind_forward:
                    # 已进入盲走模式，继续循环
                    dt = time.perf_counter() - t0
                    if dt < interval:
                        time.sleep(interval - dt)
                    continue
                else:
                    # tick_blind_forward 已处理停车逻辑，只需等待
                    time.sleep(interval)
                    continue

            pos = det.self_pos
            heading = math.radians(getattr(det, "self_angle", 0.0) or 0.0)

            # 2) 卡顿检测 + 重规划判断
            is_stuck = self.stuck_detector.update(pos)
            # is_stuck = False
            need_replan = (
                not current_path_world
                or is_stuck
                or current_target_idx >= len(current_path_world) - 1
            )

            # 3) 路径规划（若需要）
            if need_replan:
                self.move.stop()
                
                # 卡顿脱困处理：倒退 + 可选随机转向
                if is_stuck:
                    self.stuck_detector.incrementStuckCount()
                    stuck_count = self.stuck_detector.getStuckCount()
                    
                    # 连续卡顿多次后，增加随机转向
                    if stuck_count >= self.max_stuck_count_:
                        turn_bias = random.uniform(-0.6, 0.6)
                        logger.warning(
                            f"连续卡顿 {stuck_count} 次，执行随机转向脱困 "
                            f"(turn_bias={turn_bias:.2f})"
                        )
                    else:
                        turn_bias = 0.0
                        logger.info(f"检测到卡顿（第 {stuck_count} 次），执行倒退脱困")
                    
                    # 执行倒退脱困（阻塞）
                    self.move.reverse(
                        duration_s=self.reverse_duration_s_,
                        turn_bias=turn_bias
                    )
                
                grid_path, world_path = self.planner_service.plan_path(det)

                if not world_path:
                    self.move.stop()
                    time.sleep(interval)
                    continue

                current_path_grid = grid_path
                current_path_world = world_path
                current_target_idx = 0

                try:
                    self.data_hub.set_current_path(
                        grid_path=grid_path,
                        world_path=world_path,
                        target_idx=0,
                    )
                except Exception:
                    pass

                self.stuck_detector.reset()
                
                # 成功规划新路径后，如果不是卡顿触发的重规划，重置卡顿计数
                if not is_stuck:
                    self.stuck_detector.resetStuckCount()

                logger.info(f"新路径规划节点数：{len(world_path)}")

            # 4) 路径跟随：计算下一控制目标
            follow_result = self.path_follower_wrapper.follow(
                current_pos=pos,
                path_world=current_path_world,
                current_target_idx=current_target_idx,
            )

            target_world = follow_result.target_world
            dev = follow_result.distance_to_path
            dist_goal = follow_result.distance_to_goal
            goal_reached = follow_result.goal_reached
            current_target_idx = follow_result.current_idx

            # ---- 调试 UI：更新导航状态 + 路径 ----
            if self.view is not None:
                try:
                    goal_pos = getattr(det, "goal_pos", None)  # 或者从 cfg / planner_service 拿
                    self.view.update_nav_state(
                        self_pos_mmap=pos,
                        heading_rad=heading,
                        goal_pos_mmap=goal_pos,
                        path_world_mmap=current_path_world,
                        path_grid=current_path_grid,
                        target_idx=current_target_idx,
                        is_stuck=is_stuck,
                        path_deviation=dev,
                        distance_to_goal=dist_goal,
                        goal_reached=goal_reached,
                    )
                except Exception:
                    logger.exception("view.update_nav_state 失败")

            # 5) 更新 DataHub 状态
            try:
                self.data_hub.set_nav_status(
                    is_stuck=is_stuck,
                    stuck_frames=self.stuck_detector.stuck_frames,
                    path_deviation=dev,
                    distance_to_goal=dist_goal,
                    goal_reached=goal_reached,
                )

                self.data_hub.set_current_path(
                    grid_path=current_path_grid,
                    world_path=current_path_world,
                    target_idx=current_target_idx,
                )
            except Exception:
                pass

            # 6) 终点判断
            if goal_reached or target_world is None:
                self.move.stop()

                current_path_grid = []
                current_path_world = []
                current_target_idx = 0

                time.sleep(interval)
                continue

            # 7) 控制执行
            self.move.goto(
                follow_result=follow_result,
                current_pos=pos,
                heading=heading,
            )

            dt = time.perf_counter() - t0
            if dt < interval:
                time.sleep(interval - dt)

        logger.info("控制线程退出")

if __name__ == "__main__":
    """使用 F9/F10 控制导航 Runtime 的启动和停止

    - F9: 启动（若未运行则创建并启动一个 NavigationRuntime 实例）
    - F10: 停止（若正在运行则停止当前实例）
    - ESC: 退出程序（如果在运行则先停止再退出）
    - tank 进入战斗后调用该测试
    """
    from pynput import keyboard

    runtime_holder = {"rt": None}
    import os
    logger.info(f"当前工作目录: {os.getcwd()}")

    def start_runtime():
        rt = runtime_holder["rt"]
        if rt is not None and rt.is_running():
            logger.info("NavigationRuntime 已在运行，忽略 F9")
            return
        rt = NavigationRuntime()
        if rt.start(map_name="北欧峡湾"):
            runtime_holder["rt"] = rt
            logger.info("F9: NavigationRuntime 已启动")
        else:
            logger.error("F9: NavigationRuntime 启动失败")

    def stop_runtime():
        rt = runtime_holder["rt"]
        if rt is None or not rt.is_running():
            logger.info("NavigationRuntime 未运行，忽略 F10")
            return
        rt.stop()
        logger.info("F10: NavigationRuntime 已停止")

    def on_press(key):
        if key == keyboard.Key.f9:
            start_runtime()
        elif key == keyboard.Key.f10:
            stop_runtime()
        elif key == keyboard.Key.esc:
            # 退出前确保停止 Runtime
            stop_runtime()
            logger.info("ESC: 退出程序")
            return False  # 停止监听器，程序退出

    logger.info("按 F9 启动导航，F10 停止导航，ESC 退出程序")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    logger.info("NavigationRuntime 退出")
