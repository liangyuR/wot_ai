#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation runtime without UI thread manager.
Capture + detection run on dedicated threads; control loop runs on a fixed tick.
"""

import gc
import math
import random
import threading
import time
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger

from src.gui.debug_view import DpgNavDebugView
from src.navigation.controller.movement_service import MovementService
from src.navigation.nav_runtime.path_follower_wrapper import PathFollowerWrapper
from src.navigation.nav_runtime.path_planner_wrapper import PathPlannerWrapper
from src.navigation.nav_runtime.stuck_detector import StuckDetector
from src.navigation.path_planner.path_planning_service import PathPlanningService
from src.navigation.service.capture_service import CaptureService
from src.navigation.service.data_hub import DataHub
from src.utils.global_path import GetGlobalConfig
from src.vision.map_name_detector import MapNameDetector
from src.vision.minimap_anchor_detector import MinimapAnchorDetector
from src.vision.minimap_detector import MinimapDetector


class FrameBuffer:
    """Thread-safe latest-frame buffer (single producer, single consumer)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._frame_event = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        """Write the latest frame (capture thread)."""
        with self._lock:
            self._frame = frame
            self._timestamp = time.perf_counter()
            self._frame_event.set()

    def get(self, timeout: float = 0.05) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame and timestamp (detection thread)."""
        if not self._frame_event.wait(timeout):
            return None, self._timestamp
        with self._lock:
            frame = self._frame
            ts = self._timestamp
            self._frame_event.clear()
            return frame, ts


class NavigationRuntime:
    """Orchestrates capture, detection, and control loops."""

    def __init__(self):
        self.cfg = GetGlobalConfig()

        self.capture = CaptureService(self.cfg.monitor_index)
        self.view = DpgNavDebugView() if self.cfg.ui.enable else None
        self.planner_service = PathPlanningService(self.cfg)
        self.data_hub = DataHub()

        self.minimap_detector = MinimapDetector()
        self.minimap_anchor_detector = MinimapAnchorDetector()
        self.minimap_name_detector = MapNameDetector()
        self.minimap_region: Optional[dict] = None

        self.move = MovementService()
        self.path_follower_wrapper = PathFollowerWrapper()
        self.stuck_detector = StuckDetector()

        self.reverse_duration_s_ = self.cfg.stuck_detection.reverse_duration_s
        self.max_stuck_count_ = self.cfg.stuck_detection.max_stuck_count

        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._det_thread: Optional[threading.Thread] = None
        self._ctrl_thread: Optional[threading.Thread] = None
        self._view_thread: Optional[threading.Thread] = None
        self._frame_buffer: Optional[FrameBuffer] = None

    def start(self, map_name: Optional[str] = None) -> bool:
        """Start navigation runtime."""
        if self._running:
            logger.warning("NavigationRuntime already running")
            return False

        # Cleanup old resources to avoid CUDA memory leak
        self._cleanup_resources()

        # Recreate detectors and view
        self.minimap_detector = MinimapDetector()
        self.stuck_detector = StuckDetector()

        if not self.minimap_detector.LoadModel():
            logger.error("Failed to load minimap model")
            return False

        first_frame = self.capture.grab()
        if first_frame is None:
            logger.error("Initial screen grab failed; cannot detect minimap")
            return False

        self.minimap_region = self.minimap_anchor_detector.DetectRegion(first_frame)
        if not self.minimap_region:
            logger.error("Failed to detect minimap region")
            return False
        logger.info(f"Minimap region detected: {self.minimap_region}")

        if not map_name:
            from src.utils.key_controller import KeyController

            key_controller = KeyController()
            key_controller.press("b")
            time.sleep(2)
            map_name = self.minimap_name_detector.detect()
            key_controller.release("b")

            if not map_name:
                logger.error("Failed to detect map name")
                return False
            logger.info(f"Detected map name: {map_name}")

        if not self.planner_service.load_map(
            map_name, (self.minimap_region["width"], self.minimap_region["height"])
        ):
            logger.error("Cannot load map for planner")
            return False
        logger.info(f"Loaded map: {map_name}")

        if self.view is not None:
            self.view.set_grid_mask(self.planner_service.get_grid_mask())

        self._running = True
        self._frame_buffer = FrameBuffer()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._det_thread = threading.Thread(target=self._det_loop, daemon=True)
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)

        self._capture_thread.start()
        self._det_thread.start()
        self._ctrl_thread.start()

        if self.view is not None:
            self._view_thread = threading.Thread(target=self.view.run, daemon=True)
            self._view_thread.start()

        logger.info("NavigationRuntime started (capture + detect + control)")
        return True

    def stop(self) -> None:
        """Stop navigation runtime."""
        if not self._running:
            return

        logger.info("Stopping NavigationRuntime ...")
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

        try:
            self.move.stop()
        except Exception:
            pass

        self._cleanup_resources()
        logger.info("NavigationRuntime stopped")

    def _cleanup_resources(self) -> None:
        """Cleanup all resources to release CUDA memory."""
        # 1. Cleanup minimap_detector YOLO models
        if hasattr(self, "minimap_detector") and self.minimap_detector is not None:
            try:
                self.minimap_detector.Cleanup()
            except Exception as e:
                logger.debug(f"Cleanup minimap_detector error: {e}")
            self.minimap_detector = None

        # 2. Cleanup stuck_detector
        if hasattr(self, "stuck_detector") and self.stuck_detector is not None:
            self.stuck_detector = None

        # 4. Force GC and CUDA cache cleanup (only once after all engines cleaned)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("CUDA cache cleared")

    def is_running(self) -> bool:
        return self._running

    def _capture_loop(self) -> None:
        logger.info("Capture thread started")

        m = self.minimap_region
        x, y, w, h = m["x"], m["y"], m["width"], m["height"]

        while self._running:
            try:
                frame = self.capture.grab_region(x, y, w, h)
                if frame is not None:
                    self._frame_buffer.put(frame)
            except Exception as e:
                logger.error(f"Capture thread error: {e}")
                time.sleep(0.01)

        logger.info("Capture thread exited")

    def _det_loop(self) -> None:
        logger.info("Detection thread started")

        # FPS 限制
        det_fps = self.cfg.detection.max_fps
        min_interval = 1.0 / det_fps

        view_update_interval = 0  # TODO(@liangyu) 暂时不限制更新
        view_update_counter = 0
        last_processed_ts = 0.0

        while self._running:
            loop_start = time.perf_counter()
            try:
                frame, ts = self._frame_buffer.get(timeout=0.05)
                if frame is None or ts <= last_processed_ts:
                    continue
                last_processed_ts = ts

                view_update_counter += 1
                if self.view is not None and view_update_counter >= view_update_interval:
                    self.view.update_minimap_frame(frame)
                    view_update_counter = 0

                det = self.minimap_detector.Detect(frame)
                if det is None or getattr(det, "self_pos", None) is None:
                    continue

                self.data_hub.set_latest_detection(det)

            except Exception as e:
                logger.error(f"Detection thread error: {e}")
                time.sleep(0.01)
            finally:
                # FPS 限速
                elapsed = time.perf_counter() - loop_start
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

        logger.info("Detection thread exited")

    def _ctrl_loop(self) -> None:
        logger.info("Control thread started")

        ctrl_fps = 30
        interval = 1.0 / ctrl_fps

        if not self.minimap_region:
            logger.error("minimap_region missing, control thread exit")
            return

        grid_w, grid_h = self.cfg.grid.size
        mmap_w = self.minimap_region["width"]
        mmap_h = self.minimap_region["height"]
        sx = mmap_w / grid_w
        sy = mmap_h / grid_h

        self.path_planner = PathPlannerWrapper(
            planner=self.planner_service,
            minimap_size=(mmap_w, mmap_h),
            scale_xy=(sx, sy),
        )

        current_path_grid = []
        current_path_world = []
        current_target_idx = 0
        last_published_idx = -1

        view_update_interval = 3
        view_update_counter = 0

        while self._running:
            t0 = time.perf_counter()

            det = self.data_hub.get_latest_detection(max_age=1.0)
            has_detection = det is not None and getattr(det, "self_pos", None) is not None

            self.move.update_detection_status(has_detection)

            if not has_detection:
                should_blind_forward = self.move.tick_blind_forward()
                if should_blind_forward:
                    dt = time.perf_counter() - t0
                    if dt < interval:
                        time.sleep(interval - dt)
                    continue
                time.sleep(interval)
                continue

            pos = det.self_pos
            heading = math.radians(getattr(det, "self_angle", 0.0) or 0.0)

            is_stuck = self.stuck_detector.update(pos)
            need_replan = (
                not current_path_world
                or is_stuck
                or current_target_idx >= len(current_path_world) - 1
            )

            if need_replan:
                self.move.stop()

                if is_stuck:
                    self.stuck_detector.incrementStuckCount()
                    stuck_count = self.stuck_detector.getStuckCount()

                    if stuck_count >= self.max_stuck_count_:
                        turn_bias = random.uniform(-0.6, 0.6)
                        logger.warning(
                            f"Stuck {stuck_count} times, perform random turn "
                            f"(turn_bias={turn_bias:.2f})"
                        )
                    else:
                        turn_bias = 0.0
                        logger.info(f"Stuck detected #{stuck_count}, reversing to escape")

                    self.move.reverse(
                        duration_s=self.reverse_duration_s_,
                        turn_bias=turn_bias,
                    )

                grid_path, world_path = self.planner_service.plan_path(det)

                if not world_path:
                    self.move.stop()
                    time.sleep(interval)
                    continue

                current_path_grid = grid_path
                current_path_world = world_path
                current_target_idx = 0
                last_published_idx = -1

                try:
                    self.data_hub.set_current_path(
                        grid_path=grid_path,
                        world_path=world_path,
                        target_idx=0,
                    )
                except Exception:
                    pass

                self.stuck_detector.reset()
                if not is_stuck:
                    self.stuck_detector.resetStuckCount()

                logger.info(f"New path planned, nodes: {len(world_path)}")

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

            view_update_counter += 1
            if self.view is not None and view_update_counter >= view_update_interval:
                view_update_counter = 0
                try:
                    goal_pos = getattr(det, "goal_pos", None)
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
                    pass

            try:
                self.data_hub.set_nav_status(
                    is_stuck=is_stuck,
                    path_deviation=dev,
                    distance_to_goal=dist_goal,
                    goal_reached=goal_reached,
                )

                if current_target_idx != last_published_idx:
                    last_published_idx = current_target_idx
                    self.data_hub.set_current_path(
                        grid_path=current_path_grid,
                        world_path=current_path_world,
                        target_idx=current_target_idx,
                    )
            except Exception:
                pass

            if goal_reached or target_world is None:
                self.move.stop()
                current_path_grid = []
                current_path_world = []
                current_target_idx = 0
                last_published_idx = -1
                time.sleep(interval)
                continue

            self.move.goto(
                follow_result=follow_result,
                current_pos=pos,
                heading=heading,
            )

            dt = time.perf_counter() - t0
            if dt < interval:
                time.sleep(interval - dt)

        logger.info("Control thread exited")


'''
测试Demo
使用说明：
F9: 启动导航运行时
F10: 停止导航运行时
ESC: 停止并退出
'''
if __name__ == "__main__":
    import os
    from pynput import keyboard

    runtime_holder = {"rt": None}
    logger.info(f"Current working directory: {os.getcwd()}")

    def start_runtime():
        rt = runtime_holder["rt"]
        if rt is not None and rt.is_running():
            logger.info("NavigationRuntime already running; ignore F9")
            return
        rt = NavigationRuntime()
        if rt.start(map_name="Harbor"):
            runtime_holder["rt"] = rt
            logger.info("F9: NavigationRuntime started")
        else:
            logger.error("F9: NavigationRuntime failed to start")

    def stop_runtime():
        rt = runtime_holder["rt"]
        if rt is None or not rt.is_running():
            logger.info("NavigationRuntime not running; ignore F10")
            return
        rt.stop()
        logger.info("F10: NavigationRuntime stopped")

    def on_press(key):
        if key == keyboard.Key.f9:
            start_runtime()
        elif key == keyboard.Key.f10:
            stop_runtime()
        elif key == keyboard.Key.esc:
            stop_runtime()
            logger.info("ESC: exit")
            return False

    logger.info("Hold F9 to start, F10 to stop, ESC to exit.")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    logger.info("NavigationRuntime exited")
