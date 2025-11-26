#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NavigationRuntime v1 (fixed)
两线程 + DataHub（检测线程 + 控制线程）
ThreadManager 的替代方案（不带 UI）
"""

import math
import time
import threading
from typing import Optional

from loguru import logger
from src.utils.global_path import GetConfigPath
from src.navigation.config.loader import load_config
from src.navigation.service.data_hub import DataHub
# 屏幕捕获
from src.navigation.service.capture_service import CaptureService
from src.utils.screen_action import ScreenAction
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


class NavigationRuntime:
    """导航运行时：
    - detection_thread: 捕获 + YOLO 检测（限最高 FPS）
    - control_thread: 路径规划 + 路径跟随 + 转向 + 前进（固定 tick）
    - DataHub: 共享最新检测结果和导航状态
    """

    def __init__(self):
        # 配置
        config_path = GetConfigPath()
        self.cfg = load_config(config_path)

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
        angle_cfg = self.cfg.angle_detection
        if angle_cfg is not None:
            self.minimap_detector = MinimapDetector(
                model_path=self.cfg.model.path,
                conf_threshold=self.cfg.model.conf_threshold,
                iou_threshold=self.cfg.model.iou_threshold,
                smoothing_alpha=angle_cfg.smoothing_alpha,
                max_step_deg=angle_cfg.max_step_deg,
                min_area_ratio=angle_cfg.min_area_ratio,
                max_area_ratio=angle_cfg.max_area_ratio,
                min_aspect_ratio=angle_cfg.min_aspect_ratio,
                max_aspect_ratio=angle_cfg.max_aspect_ratio,
            )
        else:
            # 向后兼容：如果没有配置，使用默认值
            self.minimap_detector = MinimapDetector(
                model_path=self.cfg.model.path,
                conf_threshold=self.cfg.model.conf_threshold,
                iou_threshold=self.cfg.model.iou_threshold,
            )
        self.minimap_anchor_detector = MinimapAnchorDetector()
        self.minimap_name_detector = MapNameDetector()
        self.minimap_region: Optional[dict] = None

        # 控制与跟随
        self.move = MovementService(
            angle_dead_zone_deg=self.cfg.control.angle_dead_zone_deg,
            angle_slow_turn_deg=self.cfg.control.angle_slow_turn_deg,
            distance_stop_threshold=self.cfg.control.distance_stop_threshold,
            slow_down_distance=self.cfg.control.slow_down_distance,
            max_forward_speed=self.cfg.control.max_forward_speed,
            min_forward_factor=self.cfg.control.min_forward_factor,
            large_angle_threshold_deg=self.cfg.control.large_angle_threshold_deg,
            large_angle_speed_reduction=self.cfg.control.large_angle_speed_reduction,
            corridor_ref_width=self.cfg.control.corridor_ref_width,
            k_lat_normal=self.cfg.control.k_lat_normal,
            k_lat_edge=self.cfg.control.k_lat_edge,
            k_lat_recenter=self.cfg.control.k_lat_recenter,
            straight_angle_enter_deg=self.cfg.control.straight_angle_enter_deg,
            straight_angle_exit_deg=self.cfg.control.straight_angle_exit_deg,
            straight_lat_enter=self.cfg.control.straight_lat_enter,
            straight_lat_exit=self.cfg.control.straight_lat_exit,
            edge_speed_reduction=self.cfg.control.edge_speed_reduction,
            recenter_speed_reduction=self.cfg.control.recenter_speed_reduction,
            debug_log_interval=self.cfg.control.debug_log_interval,
            smoothing_alpha=self.cfg.control.smoothing_alpha,
            turn_deadzone=self.cfg.control.turn_deadzone,
            min_hold_time_ms=self.cfg.control.min_hold_time_ms,
            forward_hysteresis_on=self.cfg.control.forward_hysteresis_on,
            forward_hysteresis_off=self.cfg.control.forward_hysteresis_off,
        )

        self.path_follower_wrapper = PathFollowerWrapper(
            deviation_tolerance=self.cfg.control.path_deviation_tolerance,
            goal_arrival_threshold=self.cfg.control.goal_arrival_threshold,
            max_lateral_error=self.cfg.control.max_lateral_error,
            lookahead_distance=self.cfg.control.lookahead_distance,
            waypoint_switch_radius=self.cfg.control.waypoint_switch_radius,
        )

        # 卡顿检测（用配置参数初始化）
        self.stuck_detector = StuckDetector(
            move_threshold=self.cfg.control.stuck_threshold,
            frame_threshold=self.cfg.control.stuck_frames_threshold,
        )

        # 线程
        self._running = False
        self._det_thread: Optional[threading.Thread] = None
        self._ctrl_thread: Optional[threading.Thread] = None

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
        first_frame = self.capture.grab()
        if first_frame is None:
            logger.error("首次抓取屏幕失败，无法检测小地图")
            return False

        self.minimap_region = self.minimap_anchor_detector.DetectRegion(first_frame)
        if not self.minimap_region:
            logger.error("无法检测到小地图位置")
            return False
        logger.info(f"检测到小地图区域: {self.minimap_region}")

        # 检测当前地图名称，读取 mask 图片
        # V1
        # if not map_name:
        #     map_name_frame = ScreenAction().screenshot_with_key_hold('b')
        #     if map_name_frame is None:
        #         logger.error("无法截取地图名称界面")
        #         return False
        #     map_name = self.minimap_name_detector.detect(map_name_frame)
        #     if not map_name:
        #         logger.error("无法识别地图名称")
        #         return False
        #     logger.info(f"当前地图名称: {map_name}")
        # V2
        if not map_name:
            from src.utils.key_controller import KeyController
            key_controller = KeyController()
            key_controller.press('b')
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

        self._det_thread = threading.Thread(target=self._det_loop, daemon=True)
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)

        self._det_thread.start()
        self._ctrl_thread.start()

        if self.view is not None:
            self.view.run()

        logger.info("NavigationRuntime 已启动（检测 + 控制）")
        return True

    def stop(self) -> None:
        """停止导航 Runtime"""
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

    def is_running(self) -> bool:
        return self._running

    # ============================================================
    # 检测线程：产生最新 detection（限最高 FPS）
    # ============================================================
    def _det_loop(self) -> None:
        logger.info("检测线程启动")

        # detect_fps = getattr(self.cfg.performance, "detect_fps", 30)
        detect_fps = 30
        detect_fps = max(1, int(detect_fps))
        min_interval = 1.0 / detect_fps

        m = self.minimap_region
        x, y, w, h = m["x"], m["y"], m["width"], m["height"]

        while self._running:
            t0 = time.perf_counter()
            try:
                frame = self.capture.grab_region(x, y, w, h)
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                if self.view is not None:
                    self.view.update_minimap_frame(frame)

                det = self.minimap_detector.Detect(frame)
                if det is None or getattr(det, "self_pos", None) is None:
                    continue

                self.data_hub.set_latest_detection(det)

            except Exception as e:
                logger.error(f"检测线程错误: {e}")
                import traceback
                traceback.print_exc()

            dt = time.perf_counter() - t0
            if dt < min_interval:
                time.sleep(min_interval - dt)

        logger.info("检测线程退出")

    # ============================================================
    # 控制线程：固定 tick（路径规划 + 路径跟随 + 控制执行）
    # ============================================================
    def _ctrl_loop(self) -> None:
        logger.info("控制线程启动")

        # 控制线程 FPS
        ctrl_fps = 40
        ctrl_fps = max(1, int(ctrl_fps))
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
            det = self.data_hub.get_latest_detection(max_age=0.5)
            if det is None or getattr(det, "self_pos", None) is None:
                self.move.stop()
                time.sleep(interval)
                continue

            pos = det.self_pos
            heading = math.radians(getattr(det, "self_angle", 0.0) or 0.0)

            # 2) 卡顿检测 + 重规划判断
            is_stuck = self.stuck_detector.update(pos)

            need_replan = (
                not current_path_world
                or is_stuck
                or current_target_idx >= len(current_path_world) - 1
            )

            # 3) 路径规划（若需要）
            if need_replan:
                self.move.stop()
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
            new_idx = follow_result.current_idx
            used_target_idx = follow_result.target_idx_used

            current_target_idx = new_idx

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
