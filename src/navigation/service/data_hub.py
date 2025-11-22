#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Optional
from loguru import logger

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

@dataclass
class PathSnapshot:
    grid: List[Tuple[int, int]]          # A* 网格路径
    world: List[Tuple[float, float]]     # minimap 坐标路径
    target_idx: int                      # 当前跟随到的索引

@dataclass
class NavStatus:
    is_stuck: bool                       # 是否处于卡顿状态
    stuck_frames: int                    # 连续卡顿帧数
    path_deviation: float                # 当前偏离路径的距离
    distance_to_goal: float              # 距终点距离
    goal_reached: bool                   # 是否已到终点

class DataHub:
    """
    DataHub：导航系统的数据总线（第一版）
    - 保存 detection 最新帧
    - 线程安全读写
    - 提供 max_age 检查，避免旧帧污染控制逻辑
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_det = None
        self._latest_ts = 0.0

        self._path: Optional[PathSnapshot] = None
        self._nav_status: Optional[NavStatus] = None

    # --------------------------------------------------------
    # 写入最新检测（由检测线程调用）
    # --------------------------------------------------------
    def set_latest_detection(self, det) -> None:
        """
        写入最新 YOLO + minimap 检测结果
        """
        if det is None:
            return
        now = time.perf_counter()
        with self._lock:
            self._latest_det = det
            self._latest_ts = now

    # --------------------------------------------------------
    # 获取最新检测（由控制线程调用）
    # --------------------------------------------------------
    def get_latest_detection(self, max_age: float) -> Optional[object]:
        """
        获取最近一次检测结果（如超过 max_age 秒则返回 None）
        """
        now = time.perf_counter()
        with self._lock:
            det = self._latest_det
            ts = self._latest_ts

        if det is None:
            return None

        age = now - ts
        if age > max_age:
            logger.warning(f"DataHub: 检测帧过旧 age={age:.2f}s (max={max_age})")
            return None

        return det

   # ---------------- 路径 ----------------
    def set_current_path(
        self,
        grid_path: list[tuple[int, int]],
        world_path: list[tuple[float, float]],
        target_idx: int = 0,
    ) -> None:
        snap = PathSnapshot(grid=grid_path, world=world_path, target_idx=target_idx)
        with self._lock:
            self._path = snap

    def get_current_path(self) -> Optional[PathSnapshot]:
        with self._lock:
            return self._path

    # ---------------- 导航状态 ----------------
    def set_nav_status(
        self,
        is_stuck: bool,
        stuck_frames: int,
        path_deviation: float,
        distance_to_goal: float,
        goal_reached: bool,
    ) -> None:
        status = NavStatus(
            is_stuck=is_stuck,
            stuck_frames=stuck_frames,
            path_deviation=path_deviation,
            distance_to_goal=distance_to_goal,
            goal_reached=goal_reached,
        )
        with self._lock:
            self._nav_status = status

    def get_nav_status(self) -> Optional[NavStatus]:
        with self._lock:
            return self._nav_status