#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main window using DearPyGui."""

import sys
import os
import shutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from loguru import logger

from src.core.task_manager import TaskManager
from src.utils.global_path import GetVehicleScreenshotsDir, GetLogDir, GetConfigPath


class MainWindow:
    """Main control window using DearPyGui."""

    def __init__(self):
        self._title = "TANK ROBOT"
        self._window_size = (480, 820)

        # State
        self._is_running = False
        self._run_hours = 4
        self._auto_stop = False
        self._auto_shutdown = False
        self._silver_reserve = False
        self._start_time: Optional[datetime] = None

        # Vehicle images: {filename: (texture_id, thumb_w, thumb_h)}
        self._vehicle_textures: Dict[str, Tuple[int, int, int]] = {}
        self._vehicle_screenshot_dir = self._initVehicleScreenshotDir()

        # TaskManager
        self._task_manager: Optional[TaskManager] = None
        self._task_thread: Optional[threading.Thread] = None

        # DearPyGui IDs
        self._tex_registry: Optional[int] = None
        self._status_text_id: Optional[int] = None
        self._end_time_text_id: Optional[int] = None
        self._vehicle_group_id: Optional[int] = None
        self._file_dialog_id: Optional[int] = None
        self._confirm_dialog_id: Optional[int] = None
        self._confirm_callback_data: Optional[Path] = None

        # InitLog
        self._initLog()

    def _initLog(self) -> None:
        """Initialize logging configuration."""
        log_dir = GetLogDir()
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.remove()

        # Console handler
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>"
        )
        
        # DEBUG file handler - only DEBUG level
        debug_log_file = log_dir / "wot_ai_debug_{time:YYYY-MM-DD}.log"
        logger.add(
            str(debug_log_file),
            rotation="00:00",  # New file every day at midnight
            retention="10 days",  # Keep logs for 10 days
            level="DEBUG",
            filter=lambda record: record["level"].name == "DEBUG",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
        
        # INFO file handler - INFO and above (INFO, WARNING, ERROR, CRITICAL)
        info_log_file = log_dir / "wot_ai_info_{time:YYYY-MM-DD}.log"
        logger.add(
            str(info_log_file),
            rotation="00:00",  # New file every day at midnight
            retention="10 days",  # Keep logs for 10 days
            level="INFO",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} - {message}"
        )

        logger.info(f"Log initialized, saving to: {log_dir}")

    def _initVehicleScreenshotDir(self) -> Path:
        """Initialize vehicle screenshot directory."""
        default_dir = GetVehicleScreenshotsDir()
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir

    # -------------------------------------------------------------------------
    # UI Building
    # -------------------------------------------------------------------------

    def _buildUI(self) -> None:
        """Build the main UI layout."""
        win_w, win_h = self._window_size

        with dpg.window(label=self._title, width=win_w, height=win_h, no_close=True,
                        no_collapse=True, no_resize=True, no_move=True, tag="main_window"):

            # === Tips Section ===
            with dpg.collapsing_header(label="Tips", default_open=True):
                dpg.add_text("* Set game to windowed 1920x1080")
                dpg.add_text("* Recommended graphics: Medium-Low")
                dpg.add_text("* Screenshot vehicle cards, name as 1.png, 2.png...")
                dpg.add_text("* Higher priority vehicles are selected first")
                dpg.add_text("* Start from garage screen")

            dpg.add_spacer(height=5)

            # === Control Section ===
            with dpg.collapsing_header(label="Control", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Start", width=100, callback=self._onStart)
                    dpg.add_button(label="Stop", width=100, callback=self._onStop)

                dpg.add_spacer(height=5)
                self._status_text_id = dpg.add_text("Status: Stopped", color=(255, 100, 100))

            dpg.add_spacer(height=5)

            # === Duration & End Behavior Section ===
            with dpg.collapsing_header(label="Duration & End Behavior (Not Implemented)",
                                       default_open=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("Run duration limit:")
                    dpg.add_input_int(
                        default_value=self._run_hours,
                        width=80,
                        min_value=1,
                        max_value=24,
                        callback=lambda s, a: setattr(self, '_run_hours', a)
                    )
                    dpg.add_text("hours")

                dpg.add_checkbox(
                    label="Auto stop when duration reached",
                    default_value=self._auto_stop,
                    callback=lambda s, a: setattr(self, '_auto_stop', a)
                )
                dpg.add_checkbox(
                    label="Auto shutdown when duration reached (Admin required)",
                    default_value=self._auto_shutdown,
                    callback=lambda s, a: setattr(self, '_auto_shutdown', a)
                )

                self._end_time_text_id = dpg.add_text("Estimated end time: --")

            dpg.add_spacer(height=5)

            # === Feature Expansion Section ===
            with dpg.collapsing_header(label="Feature Expansion (Not Implemented)",
                                       default_open=False):
                dpg.add_checkbox(
                    label="Enable silver reserve on start",
                    default_value=self._silver_reserve,
                    callback=lambda s, a: setattr(self, '_silver_reserve', a)
                )

            dpg.add_spacer(height=5)

            # === Vehicle Priority Section ===
            with dpg.collapsing_header(label="Vehicle Priority", default_open=True):
                dpg.add_text("Select vehicles in priority order:")
                dpg.add_text("1.png -> 2.png -> 3.png ...", color=(150, 150, 150))

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add Screenshot", callback=self._onAddScreenshot)
                    dpg.add_button(label="Open Screenshot Folder",
                                   callback=self._onOpenScreenshotDir)
                    dpg.add_button(label="Refresh", callback=self._refreshVehicleList)

                dpg.add_spacer(height=5)
                dpg.add_text("Current vehicle priority:")

                # Scrollable vehicle list container
                with dpg.child_window(height=200, border=True, horizontal_scrollbar=True):
                    self._vehicle_group_id = dpg.add_group(horizontal=True)

            dpg.add_spacer(height=10)

            # === Footer ===
            with dpg.group(horizontal=True):
                dpg.add_text("Version: v0.1.0", color=(128, 128, 128))
                dpg.add_button(label="Open Log Folder", callback=self._onOpenLogDir)
                dpg.add_button(label="Open Config", callback=self._onOpenConfigDir)

        # === File Dialog ===
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._onFileSelected,
            cancel_callback=lambda: None,
            width=500,
            height=400,
            tag="file_dialog"
        ):
            dpg.add_file_extension(".png", color=(0, 255, 0))
            dpg.add_file_extension(".*")

        self._file_dialog_id = dpg.last_item()

        # === Confirm Delete Dialog ===
        with dpg.window(
            label="Confirm Delete",
            modal=True,
            show=False,
            no_resize=True,
            width=300,
            height=100,
            tag="confirm_dialog"
        ):
            dpg.add_text("Are you sure you want to delete this screenshot?")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Yes", width=100, callback=self._onConfirmDelete)
                dpg.add_button(label="No", width=100,
                               callback=lambda: dpg.configure_item("confirm_dialog", show=False))

        self._confirm_dialog_id = dpg.last_item()

    # -------------------------------------------------------------------------
    # Vehicle List Management
    # -------------------------------------------------------------------------

    def _refreshVehicleList(self, sender=None, app_data=None) -> None:
        """Refresh the vehicle thumbnail list."""
        if self._vehicle_group_id is None:
            return

        # Clear existing items
        dpg.delete_item(self._vehicle_group_id, children_only=True)

        # Clear old textures
        for filename, (tex_id, _, _) in self._vehicle_textures.items():
            try:
                dpg.delete_item(tex_id)
            except Exception:
                pass
        self._vehicle_textures.clear()

        # Load images from directory
        if not self._vehicle_screenshot_dir.exists():
            return

        image_files = sorted(self._vehicle_screenshot_dir.glob("*.png"))

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Create thumbnail
                h, w = img.shape[:2]
                max_thumb_size = 80
                scale = min(max_thumb_size / w, max_thumb_size / h, 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                thumbnail = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Convert to RGBA float
                rgba_data = self._bgrToRgbaFloat(thumbnail)

                # Create dynamic texture
                tex_id = dpg.add_dynamic_texture(
                    new_w, new_h, rgba_data, parent=self._tex_registry
                )
                self._vehicle_textures[img_path.name] = (tex_id, new_w, new_h)

                # Create item group
                with dpg.group(parent=self._vehicle_group_id):
                    dpg.add_image(tex_id)
                    dpg.add_text(img_path.name, color=(200, 200, 200))
                    dpg.add_button(
                        label="Delete",
                        callback=lambda s, a, p=img_path: self._onDeleteScreenshot(p),
                        small=True
                    )
                    dpg.add_spacer(width=10)

            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")

    @staticmethod
    def _bgrToRgbaFloat(frame_bgr: np.ndarray) -> List[float]:
        """Convert BGR uint8 image to RGBA float32 list in [0, 1]."""
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([frame_rgb, alpha], axis=2)
        rgba_f = (rgba.astype(np.float32) / 255.0).reshape(-1).tolist()
        return rgba_f

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def _onStart(self, sender=None, app_data=None) -> None:
        """Start task manager."""
        if self._is_running:
            logger.warning("Task already running")
            return

        self._task_manager = TaskManager(
            run_hours=self._run_hours,
            auto_stop=self._auto_stop,
            auto_shutdown=self._auto_shutdown,
        )

        self._is_running = True
        self._start_time = datetime.now()
        self._task_thread = threading.Thread(
            target=self._task_manager.run_forever, daemon=True
        )
        self._task_thread.start()

        logger.info("Task manager started")

    def _onStop(self, sender=None, app_data=None) -> None:
        """Stop task manager."""
        if not self._is_running:
            return

        self._is_running = False

        if self._task_manager:
            self._task_manager.stop()

        if self._task_thread and self._task_thread.is_alive():
            self._task_thread.join(timeout=2.0)

        self._task_manager = None
        self._start_time = None
        logger.info("Task manager stopped")

    def _onAddScreenshot(self, sender=None, app_data=None) -> None:
        """Open file dialog to add screenshot."""
        dpg.show_item("file_dialog")

    def _onFileSelected(self, sender, app_data) -> None:
        """Handle file selection from dialog."""
        if not app_data or "file_path_name" not in app_data:
            return

        file_path = app_data["file_path_name"]
        if not file_path:
            return

        try:
            src_path = Path(file_path)
            dest_path = self._vehicle_screenshot_dir / src_path.name
            shutil.copy2(src_path, dest_path)
            self._refreshVehicleList()
            logger.info(f"Added vehicle screenshot: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to add screenshot: {e}")

    def _onDeleteScreenshot(self, img_path: Path) -> None:
        """Show delete confirmation dialog."""
        self._confirm_callback_data = img_path
        dpg.configure_item("confirm_dialog", show=True)

    def _onConfirmDelete(self, sender=None, app_data=None) -> None:
        """Handle delete confirmation."""
        dpg.configure_item("confirm_dialog", show=False)

        if self._confirm_callback_data is None:
            return

        img_path = self._confirm_callback_data
        self._confirm_callback_data = None

        try:
            if img_path.exists():
                img_path.unlink()
                self._refreshVehicleList()
                logger.info(f"Deleted vehicle screenshot: {img_path}")
        except Exception as e:
            logger.error(f"Failed to delete screenshot: {e}")

    def _onOpenScreenshotDir(self, sender=None, app_data=None) -> None:
        """Open vehicle screenshot directory."""
        try:
            if sys.platform == "win32":
                os.startfile(str(self._vehicle_screenshot_dir))
        except Exception as e:
            logger.error(f"Failed to open directory: {e}")

    def _onOpenLogDir(self, sender=None, app_data=None) -> None:
        """Open log directory."""
        try:
            if sys.platform == "win32":
                os.startfile(str(GetLogDir()))
        except Exception as e:
            logger.error(f"Failed to open log directory: {e}")

    def _onOpenConfigDir(self, sender=None, app_data=None) -> None:
        """Open config file location."""
        try:
            if sys.platform == "win32":
                os.startfile(str(GetConfigPath()))
        except Exception as e:
            logger.error(f"Failed to open config: {e}")

    # -------------------------------------------------------------------------
    # Status Update
    # -------------------------------------------------------------------------

    def _updateStatus(self) -> None:
        """Update status display on each frame."""
        if self._status_text_id is None:
            return

        if self._is_running:
            if self._start_time and self._run_hours > 0:
                end_time = self._start_time + timedelta(hours=self._run_hours)
                remaining = end_time - datetime.now()

                if remaining.total_seconds() > 0:
                    hours = int(remaining.total_seconds() // 3600)
                    minutes = int((remaining.total_seconds() % 3600) // 60)
                    seconds = int(remaining.total_seconds() % 60)
                    status_text = f"Status: Running (Remaining {hours:02d}:{minutes:02d}:{seconds:02d})"
                    end_text = f"Estimated end time: {end_time:%Y-%m-%d %H:%M}"
                else:
                    status_text = "Status: Running (Time exceeded)"
                    end_text = "Estimated end time: --"

                dpg.set_value(self._status_text_id, status_text)
                dpg.configure_item(self._status_text_id, color=(100, 255, 100))

                if self._end_time_text_id:
                    dpg.set_value(self._end_time_text_id, end_text)
            else:
                dpg.set_value(self._status_text_id, "Status: Running")
                dpg.configure_item(self._status_text_id, color=(100, 255, 100))
        else:
            dpg.set_value(self._status_text_id, "Status: Stopped")
            dpg.configure_item(self._status_text_id, color=(255, 100, 100))

            if self._end_time_text_id:
                dpg.set_value(self._end_time_text_id, "Estimated end time: --")

    # -------------------------------------------------------------------------
    # Main Entry
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Create DearPyGui context and start UI loop (blocking)."""
        dpg.create_context()

        # Texture registry
        self._tex_registry = dpg.add_texture_registry()

        # Build UI
        self._buildUI()

        # Load initial vehicle list
        self._refreshVehicleList()

        # Create viewport
        win_w, win_h = self._window_size
        dpg.create_viewport(title=self._title, width=win_w, height=win_h, resizable=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Set primary window
        dpg.set_primary_window("main_window", True)

        # Render loop
        while dpg.is_dearpygui_running():
            self._updateStatus()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


def main():
    try:
        window = MainWindow()
        window.run()
        return 0
    except Exception as e:
        logger.error(f"Main program error: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
