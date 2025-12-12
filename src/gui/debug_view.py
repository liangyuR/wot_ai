from __future__ import annotations

"""DearPyGui-based navigation debug view.

This module provides:
- NavDebugView: a small protocol-like interface used by nav runtime
- DpgNavDebugView: a concrete implementation using dearpygui 2.x

Typical usage (simplified):

    debug_view = DpgNavDebugView(title="WOT Nav Debug")

    # In your nav runtime:
    self.debug_view = debug_view

    # Start UI in main thread / separate process:
    debug_view.run()

Then, from detection / control threads, call:

    debug_view.update_minimap_frame(frame_bgr)
    debug_view.set_grid_mask(grid_mask)
    debug_view.update_nav_state(...)

All methods are non-blocking and thread-safe (simple mutex + state snapshot).
"""

import math
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np
import dearpygui.dearpygui as dpg


PointF = Tuple[float, float]
GridPoint = Tuple[int, int]


class NavDebugView(Protocol):
    """Abstract interface used by navigation runtime for debug visualization.

    All methods must be non-blocking and thread-safe. Implementations are
    responsible for their own rendering (DearPyGui, Qt, etc.).
    """

    def set_grid_mask(self, grid_mask: np.ndarray) -> None:
        """Set or update the grid mask map.

        grid_mask: HxW, 0/1 or float, resolution equal to cfg.grid.size.
        """

    def update_minimap_frame(self, frame_bgr: np.ndarray) -> None:
        """Update the latest minimap frame (captured by MSS).

        frame_bgr: h x w x 3, uint8, BGR (OpenCV convention).
        """

    def update_nav_state(
        self,
        self_pos_mmap: PointF,
        heading_rad: float,
        goal_pos_mmap: Optional[PointF],
        path_world_mmap: Optional[Sequence[PointF]],
        path_grid: Optional[Sequence[GridPoint]],
        target_idx: int,
        is_stuck: bool,
        path_deviation: float,
        distance_to_goal: float,
        goal_reached: bool,
    ) -> None:
        """Update current navigation state and path.

        - *_mmap: coordinates in minimap pixel space (same as det.self_pos)
        - path_world_mmap: planned path in minimap coordinates
        - path_grid: path in grid coordinates (for grid view)
        """


@dataclass
class _DebugState:
    """Internal shared state accessed by worker threads + UI thread.

    All fields are overwritten atomically under a mutex in worker threads; UI
    thread takes snapshots on each frame.
    """

    minimap_frame: Optional[np.ndarray] = None  # H x W x 3, uint8, BGR
    grid_mask: Optional[np.ndarray] = None      # Hg x Wg, 0/1 or float

    self_pos_mmap: Optional[PointF] = None
    heading_rad: float = 0.0
    goal_pos_mmap: Optional[PointF] = None

    path_world_mmap: List[PointF] = field(default_factory=list)
    path_grid: List[GridPoint] = field(default_factory=list)
    target_idx: int = 0

    is_stuck: bool = False
    path_deviation: float = 0.0
    distance_to_goal: float = 0.0
    goal_reached: bool = False


class DpgNavDebugView(NavDebugView):
    """DearPyGui implementation of NavDebugView.

    This class owns the DearPyGui context and event loop. It should be
    instantiated once and its `run()` method called from the main thread.

    Other threads (detection / control) call the NavDebugView methods to
    update the shared state; the UI thread renders a snapshot each frame.
    """

    def __init__(
        self,
        title: str = "Nav Debug",
        window_size: Tuple[int, int] = (960, 720),
        enable: bool = True,
        grid_to_minimap_h: Optional[np.ndarray] = None,   # 新增: homography
    ) -> None:
        self._title = title
        self._window_size = window_size
        self._enable = enable

        self._state = _DebugState()
        self._lock = threading.Lock()

        # DearPyGui ids
        self._tex_registry: Optional[int] = None
        self._tex_id_minimap: Optional[int] = None
        self._tex_id_grid: Optional[int] = None

        self._minimap_tex_w: int = 2
        self._minimap_tex_h: int = 2
        self._grid_tex_w: int = 2
        self._grid_tex_h: int = 2

        # 叠加到 minimap 上用的 grid 纹理
        self._tex_id_grid_overlay: Optional[int] = None
        self._grid_overlay_tex_w: int = 2
        self._grid_overlay_tex_h: int = 2

        self._drawlist_minimap: Optional[int] = None
        self._drawlist_grid: Optional[int] = None
        self._txt_status: Optional[int] = None
        self._txt_metrics: Optional[int] = None

        # grid -> minimap 的透视变换矩阵 H (3x3)
        self._H_grid2mmap: Optional[np.ndarray] = None
        if grid_to_minimap_h is not None:
            self.set_grid_to_minimap_homography(grid_to_minimap_h)

    # ---------------------------------------------------------------------
    # NavDebugView interface (thread-safe, called from worker threads)
    # ---------------------------------------------------------------------

    def set_grid_mask(self, grid_mask: np.ndarray) -> None:
        if not self._enable:
            return
        with self._lock:
            self._state.grid_mask = grid_mask.copy()

    def update_minimap_frame(self, frame_bgr: np.ndarray) -> None:
        if not self._enable:
            return
        # store a copy to avoid lifetime issues / in-place modifications
        with self._lock:
            self._state.minimap_frame = frame_bgr.copy()

    def update_nav_state(
        self,
        self_pos_mmap: PointF,
        heading_rad: float,
        goal_pos_mmap: Optional[PointF],
        path_world_mmap: Optional[Sequence[PointF]],
        path_grid: Optional[Sequence[GridPoint]],
        target_idx: int,
        is_stuck: bool,
        path_deviation: float,
        distance_to_goal: float,
        goal_reached: bool,
    ) -> None:
        if not self._enable:
            return
        with self._lock:
            st = self._state
            st.self_pos_mmap = self_pos_mmap
            st.heading_rad = heading_rad
            st.goal_pos_mmap = goal_pos_mmap
            st.path_world_mmap = list(path_world_mmap or [])
            st.path_grid = list(path_grid or [])
            st.target_idx = target_idx
            st.is_stuck = is_stuck
            st.path_deviation = path_deviation
            st.distance_to_goal = distance_to_goal
            st.goal_reached = goal_reached

    # ---------------------------------------------------------------------
    # Extra public API
    # ---------------------------------------------------------------------

    def set_grid_to_minimap_homography(self, H: np.ndarray) -> None:
        """设置 grid 坐标到 minimap 像素坐标的 3x3 透视变换矩阵.

        一般用 cv2.getPerspectiveTransform 得到:
            H = cv2.getPerspectiveTransform(pts_grid, pts_minimap)
        """
        H = np.asarray(H, dtype=np.float32)
        if H.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {H.shape}")
        self._H_grid2mmap = H

    # ---------------------------------------------------------------------
    # Public UI entrypoint
    # ---------------------------------------------------------------------

    def _resetDpgState(self) -> None:
        """重置所有 DearPyGui 相关的 ID 和尺寸状态（支持多次启动）"""
        self._tex_registry = None
        self._tex_id_minimap = None
        self._tex_id_grid = None
        self._tex_id_grid_overlay = None
        self._minimap_tex_w = 2
        self._minimap_tex_h = 2
        self._grid_tex_w = 2
        self._grid_tex_h = 2
        self._grid_overlay_tex_w = 2
        self._grid_overlay_tex_h = 2
        self._drawlist_minimap = None
        self._drawlist_grid = None
        self._txt_status = None
        self._txt_metrics = None

    def run(self) -> None:
        """Create DearPyGui context and start UI loop (blocking).

        This should be called from the main thread. Other threads can safely
        call the NavDebugView methods to update state.
        """

        if not self._enable:
            return

        # 重置所有 DearPyGui ID（支持多次启动）
        self._resetDpgState()

        dpg.create_context()

        # texture registry and placeholder textures
        self._tex_registry = dpg.add_texture_registry()

        # minimap texture placeholder
        placeholder_data = [0.0] * (2 * 2 * 4)
        self._tex_id_minimap = dpg.add_dynamic_texture(
            2,
            2,
            placeholder_data,
            parent=self._tex_registry,
        )

        # grid texture placeholder
        self._tex_id_grid = dpg.add_dynamic_texture(
            2,
            2,
            placeholder_data,
            parent=self._tex_registry,
        )

        # grid overlay texture placeholder
        self._tex_id_grid_overlay = dpg.add_dynamic_texture(
            2,
            2,
            placeholder_data,
            parent=self._tex_registry,
        )

        win_w, win_h = self._window_size

        with dpg.window(label=self._title, width=win_w, height=win_h):
            with dpg.group(horizontal=True):
                # left: minimap view
                with dpg.group():
                    dpg.add_text("Minimap")
                    # Create drawlist for minimap drawing
                    self._drawlist_minimap = dpg.add_drawlist(width=400, height=400)

                # right: grid view + text info
                with dpg.group():
                    dpg.add_text("Grid Map")
                    # Create drawlist for grid drawing
                    self._drawlist_grid = dpg.add_drawlist(width=400, height=400)
                    dpg.add_spacer(height=10)
                    self._txt_status = dpg.add_text("")
                    self._txt_metrics = dpg.add_text("")

        dpg.create_viewport(title=self._title, width=win_w, height=win_h)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Custom render loop (replaces start_dearpygui for per-frame updates)
        while dpg.is_dearpygui_running():
            self._on_render()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def close(self) -> None:
        if not self._enable:
            return
        # 避免跨线程直接销毁上下文，而是通知 UI 循环停止
        if dpg.is_dearpygui_running():
            dpg.stop_dearpygui()

    # ---------------------------------------------------------------------
    # Internal helpers (UI thread only)
    # ---------------------------------------------------------------------

    def _ensure_minimap_texture(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        if (
            self._tex_id_minimap is None
            or w != self._minimap_tex_w
            or h != self._minimap_tex_h
        ):
            if self._tex_id_minimap is not None:
                dpg.delete_item(self._tex_id_minimap)

            data = [0.0] * (w * h * 4)
            self._tex_id_minimap = dpg.add_dynamic_texture(
                w,
                h,
                data,
                parent=self._tex_registry,
            )
            self._minimap_tex_w = w
            self._minimap_tex_h = h

            if self._drawlist_minimap is not None:
                dpg.configure_item(self._drawlist_minimap, width=w, height=h)

    def _ensure_grid_texture(self, grid_mask: np.ndarray) -> None:
        h, w = grid_mask.shape[:2]
        if (
            self._tex_id_grid is None
            or w != self._grid_tex_w
            or h != self._grid_tex_h
        ):
            if self._tex_id_grid is not None:
                dpg.delete_item(self._tex_id_grid)

            data = [0.0] * (w * h * 4)
            self._tex_id_grid = dpg.add_dynamic_texture(
                w,
                h,
                data,
                parent=self._tex_registry,
            )
            self._grid_tex_w = w
            self._grid_tex_h = h

            if self._drawlist_grid is not None:
                dpg.configure_item(self._drawlist_grid, width=w, height=h)

    def _ensure_grid_overlay_texture(self, overlay_mask: np.ndarray) -> None:
        """确保叠加在 minimap 上的 grid 纹理尺寸匹配 overlay_mask."""
        h, w = overlay_mask.shape[:2]
        if (
            self._tex_id_grid_overlay is None
            or w != self._grid_overlay_tex_w
            or h != self._grid_overlay_tex_h
        ):
            if self._tex_id_grid_overlay is not None:
                dpg.delete_item(self._tex_id_grid_overlay)

            data = [0.0] * (w * h * 4)
            self._tex_id_grid_overlay = dpg.add_dynamic_texture(
                w,
                h,
                data,
                parent=self._tex_registry,
            )
            self._grid_overlay_tex_w = w
            self._grid_overlay_tex_h = h

    def _warp_grid_to_minimap(self, grid_mask: np.ndarray, mmap_w: int, mmap_h: int) -> np.ndarray:
        """将 grid_mask 透视/缩放到 minimap 尺寸.

        优先用 homography; 没有 H 时退化为简单 resize。
        """
        src = grid_mask.astype(np.float32)
        if self._H_grid2mmap is not None:
            warped = cv2.warpPerspective(
                src,
                self._H_grid2mmap,
                (mmap_w, mmap_h),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
        else:
            warped = cv2.resize(src, (mmap_w, mmap_h), interpolation=cv2.INTER_NEAREST)
        return warped

    @staticmethod
    def _bgr_to_rgba_float(frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR uint8 image to RGBA float32 in [0, 1], flattened.
        """

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([frame_rgb, alpha], axis=2)
        rgba_f = (rgba.astype(np.float32) / 255.0).reshape(-1)
        return rgba_f

    @staticmethod
    def _grid_mask_to_rgba_float(grid_mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Convert grid mask (0/1 or float) to RGBA float texture.

        Obstacles are darker; free space lighter.

        alpha: 透明度 0~1
        """

        g = grid_mask.astype(np.float32)
        g = np.clip(g, 0.0, 1.0)

        # Treat 1.0 as obstacle, 0.0 as free: invert for brightness
        brightness = 0.8 - 0.6 * g  # free ~0.8, obstacle ~0.2

        h, w = g.shape
        rgb = np.stack([brightness] * 3, axis=2)  # H, W, 3
        alpha_arr = np.full((h, w, 1), float(alpha), dtype=np.float32)

        rgba = np.concatenate([rgb, alpha_arr], axis=2)
        rgba_f = rgba.reshape(-1)
        return rgba_f

    # ------------------------------------------------------------------
    # Render callback (UI thread)
    # ------------------------------------------------------------------

    def _on_render(self, sender=None, app_data=None) -> None:  # type: ignore[override]
        # Take a snapshot of the state under lock
        with self._lock:
            st = self._state

            frame = st.minimap_frame
            grid_mask = st.grid_mask

            self_pos = st.self_pos_mmap
            heading = st.heading_rad
            goal_pos = st.goal_pos_mmap
            path_world = list(st.path_world_mmap)
            path_grid = list(st.path_grid)
            target_idx = st.target_idx

            is_stuck = st.is_stuck
            path_dev = st.path_deviation
            dist_goal = st.distance_to_goal
            goal_reached = st.goal_reached

        # --- Minimap texture + drawing ---
        if frame is not None and self._drawlist_minimap is not None:
            self._ensure_minimap_texture(frame)
            rgba_f = self._bgr_to_rgba_float(frame)
            dpg.set_value(self._tex_id_minimap, rgba_f.tolist())

            # Clear drawlist by deleting all children
            try:
                children = dpg.get_item_children(self._drawlist_minimap, slot=1)
                if isinstance(children, dict):
                    # If returns dict, get slot 1 items
                    children = children.get(1, [])
                for child in children:
                    dpg.delete_item(child)
            except Exception:
                # Fallback: try alternative method
                try:
                    children_dict = dpg.get_item_children(self._drawlist_minimap)
                    if isinstance(children_dict, dict) and 1 in children_dict:
                        for item in children_dict[1]:
                            dpg.delete_item(item)
                except Exception:
                    pass

            # draw background minimap image
            w = self._minimap_tex_w
            h = self._minimap_tex_h
            dpg.draw_image(
                self._tex_id_minimap,
                pmin=(0, 0),
                pmax=(w, h),
                parent=self._drawlist_minimap,
            )

            # === 将 grid 透视/缩放到 minimap，并以 30% 透明度叠加 ===
            if grid_mask is not None and self._tex_registry is not None:
                # 1) warp 或 resize 到 minimap 分辨率
                overlay_mask = self._warp_grid_to_minimap(grid_mask, w, h)

                # 2) 确保 overlay 纹理尺寸匹配
                self._ensure_grid_overlay_texture(overlay_mask)

                # 3) 转 RGBA，alpha=0.3 (约 30% 显示)
                rgba_f_grid_overlay = self._grid_mask_to_rgba_float(
                    overlay_mask,
                    alpha=0.3,
                )
                dpg.set_value(self._tex_id_grid_overlay, rgba_f_grid_overlay.tolist())

                # 4) 在 minimap drawlist 上叠加
                dpg.draw_image(
                    self._tex_id_grid_overlay,
                    pmin=(0, 0),
                    pmax=(w, h),
                    parent=self._drawlist_minimap,
                )
            # === 新增结束 ===

            # draw path (world path in minimap coords)
            if len(path_world) >= 2:
                dpg.draw_polyline(
                    points=path_world,
                    color=(0, 255, 0, 255),
                    thickness=2.0,
                    parent=self._drawlist_minimap,
                )

            # highlight current target point
            if 0 <= target_idx < len(path_world):
                tx, ty = path_world[target_idx]
                dpg.draw_circle(
                    center=(tx, ty),
                    radius=4.0,
                    color=(0, 255, 255, 255),
                    fill=(0, 255, 255, 64),
                    parent=self._drawlist_minimap,
                )

            # draw self pose as an arrow-like triangle
            if self_pos is not None:
                cx, cy = self_pos
                angle = heading
                length = 14.0
                width = 8.0

                tip = (
                    cx + length * math.cos(angle),
                    cy + length * math.sin(angle),
                )
                left = (
                    cx + width * math.cos(angle + 2.5),
                    cy + width * math.sin(angle + 2.5),
                )
                right = (
                    cx + width * math.cos(angle - 2.5),
                    cy + width * math.sin(angle - 2.5),
                )

                dpg.draw_triangle(
                    p1=tip,
                    p2=left,
                    p3=right,
                    color=(255, 255, 0, 255),
                    fill=(255, 255, 0, 128),
                    parent=self._drawlist_minimap,
                )

            # draw goal position as a cross
            if goal_pos is not None:
                gx, gy = goal_pos
                size = 6.0
                dpg.draw_line(
                    p1=(gx - size, gy - size),
                    p2=(gx + size, gy + size),
                    color=(255, 0, 0, 255),
                    thickness=2.0,
                    parent=self._drawlist_minimap,
                )
                dpg.draw_line(
                    p1=(gx - size, gy + size),
                    p2=(gx + size, gy - size),
                    color=(255, 0, 0, 255),
                    thickness=2.0,
                    parent=self._drawlist_minimap,
                )

        # --- Grid texture + drawing ---
        if grid_mask is not None and self._drawlist_grid is not None:
            self._ensure_grid_texture(grid_mask)
            rgba_f_grid = self._grid_mask_to_rgba_float(grid_mask)  # alpha 默认 1.0
            dpg.set_value(self._tex_id_grid, rgba_f_grid.tolist())

            # Clear drawlist by deleting all children
            try:
                children = dpg.get_item_children(self._drawlist_grid, slot=1)
                if isinstance(children, dict):
                    # If returns dict, get slot 1 items
                    children = children.get(1, [])
                for child in children:
                    dpg.delete_item(child)
            except Exception:
                # Fallback: try alternative method
                try:
                    children_dict = dpg.get_item_children(self._drawlist_grid)
                    if isinstance(children_dict, dict) and 1 in children_dict:
                        for item in children_dict[1]:
                            dpg.delete_item(item)
                except Exception:
                    pass

            w = self._grid_tex_w
            h = self._grid_tex_h
            dpg.draw_image(
                self._tex_id_grid,
                pmin=(0, 0),
                pmax=(w, h),
                parent=self._drawlist_grid,
            )

            # draw grid path
            if len(path_grid) >= 2:
                pts = [
                    (float(gx) + 0.5, float(gy) + 0.5) for (gx, gy) in path_grid
                ]
                dpg.draw_polyline(
                    points=pts,
                    color=(0, 255, 0, 255),
                    thickness=1.5,
                    parent=self._drawlist_grid,
                )

                if 0 <= target_idx < len(pts):
                    tx, ty = pts[target_idx]
                    dpg.draw_circle(
                        center=(tx, ty),
                        radius=2.5,
                        color=(0, 255, 255, 255),
                        fill=(0, 255, 255, 64),
                        parent=self._drawlist_grid,
                    )

        # --- Text status / metrics ---
        if self._txt_status is not None:
            lines = []
            if self_pos is not None:
                sx, sy = self_pos
                lines.append(
                    f"Self: ({sx:.1f}, {sy:.1f}), heading={math.degrees(heading):.1f}°"
                )
            if goal_pos is not None:
                gx, gy = goal_pos
                lines.append(f"Goal: ({gx:.1f}, {gy:.1f})")
            lines.append(f"Path nodes: {len(path_world)}, target_idx: {target_idx}")
            lines.append(f"Stuck: {is_stuck}, Goal reached: {goal_reached}")
            dpg.set_value(self._txt_status, "\n".join(lines))

        if self._txt_metrics is not None:
            txt = (
                f"Deviation: {path_dev:.1f} px | "
                f"Dist to goal: {dist_goal:.1f} px"
            )
            dpg.set_value(self._txt_metrics, txt)


def _demo_feeder(view: DpgNavDebugView) -> None:
    """Background thread: feeds fake data into the debug view.

    - Minimaps: simple colored background with a moving dot track
    - Grid: random obstacle map
    - Path: a simple polyline from start to goal
    """

    # Fake grid: 64x64 with random obstacles
    import time

    grid_h, grid_w = 64, 64
    grid_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
    rng = np.random.default_rng(0)
    obstacle_mask = rng.random((grid_h, grid_w)) < 0.15
    grid_mask[obstacle_mask] = 1
    view.set_grid_mask(grid_mask)

    # Fake minimap size
    mmap_w, mmap_h = 400, 400

    t = 0.0
    while True:
        # 1) build a plain minimap background
        minimap = np.zeros((mmap_h, mmap_w, 3), dtype=np.uint8)
        minimap[:, :] = (30, 30, 30)

        # draw some grid lines to make it visually easier to debug
        step = 40
        for x in range(0, mmap_w, step):
            cv2.line(minimap, (x, 0), (x, mmap_h - 1), (60, 60, 60), 1)
        for y in range(0, mmap_h, step):
            cv2.line(minimap, (0, y), (mmap_w - 1, y), (60, 60, 60), 1)

        # 2) fake path: from left-bottom to right-top along a simple curve
        num_nodes = 40
        path_world = []
        for i in range(num_nodes):
            u = i / (num_nodes - 1)
            x = 40 + u * (mmap_w - 80)
            y = mmap_h - 40 - (mmap_h - 80) * u * (0.5 + 0.5 * math.sin(2 * math.pi * u))
            path_world.append((x, y))

        # 3) position moves along the path in a loop
        u = (math.sin(t) * 0.5 + 0.5)  # 0..1
        idx_f = u * (num_nodes - 1)
        idx0 = int(idx_f)
        idx1 = min(num_nodes - 1, idx0 + 1)
        alpha = idx_f - idx0
        x0, y0 = path_world[idx0]
        x1, y1 = path_world[idx1]
        self_x = x0 * (1 - alpha) + x1 * alpha
        self_y = y0 * (1 - alpha) + y1 * alpha

        heading = math.atan2(y1 - y0, x1 - x0)

        # 4) goal: fixed at path end
        goal_x, goal_y = path_world[-1]

        # 5) grid path: simple diagonal from bottom-left to top-right
        grid_path = [(i, grid_h - 1 - i * grid_h // grid_w) for i in range(grid_w)]

        # 6) feed data into view
        view.update_minimap_frame(minimap)
        view.update_nav_state(
            self_pos_mmap=(self_x, self_y),
            heading_rad=heading,
            goal_pos_mmap=(goal_x, goal_y),
            path_world_mmap=path_world,
            path_grid=grid_path,
            target_idx=idx0,
            is_stuck=False,
            path_deviation=0.0,
            distance_to_goal=math.hypot(goal_x - self_x, goal_y - self_y),
            goal_reached=False,
        )

        t += 0.05
        time.sleep(1.0 / 30.0)  # ~30 FPS


if __name__ == "__main__":
    # Simple standalone demo: run this file directly to see the debug UI.
    debug_view = DpgNavDebugView(title="Nav Debug Demo")

    feeder_thread = threading.Thread(target=_demo_feeder, args=(debug_view,), daemon=True)
    feeder_thread.start()

    debug_view.run()
