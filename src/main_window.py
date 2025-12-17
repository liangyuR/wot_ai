#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main window using Tkinter."""

import sys
import os
import shutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from loguru import logger

from src.core.task_manager import TaskManager
from src.utils.global_path import GetVehicleScreenshotsDir, GetLogDir, GetConfigPath


class MainWindow:
    """Main control window using Tkinter."""

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

        # Vehicle images: {filename: PhotoImage}
        self._vehicle_images: Dict[str, ImageTk.PhotoImage] = {}
        self._vehicle_screenshot_dir = self._initVehicleScreenshotDir()

        # TaskManager
        self._task_manager: Optional[TaskManager] = None
        self._task_thread: Optional[threading.Thread] = None

        # Tkinter widgets
        self._root: Optional[tk.Tk] = None
        self._status_label: Optional[tk.Label] = None
        self._end_time_label: Optional[tk.Label] = None
        self._vehicle_frame: Optional[ttk.Frame] = None
        self._vehicle_canvas: Optional[tk.Canvas] = None
        self._vehicle_scrollbar: Optional[ttk.Scrollbar] = None

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
        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.geometry(f"{self._window_size[0]}x{self._window_size[1]}")
        self._root.resizable(False, False)

        # Main container with padding
        main_frame = ttk.Frame(self._root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Tips Section ===
        tips_frame = ttk.LabelFrame(main_frame, text="Tips", padding="5")
        tips_frame.pack(fill=tk.X, pady=5)

        tips_text = [
            "* Set game to windowed 1920x1080",
            "* Recommended graphics: Medium-Low",
            "* Screenshot vehicle cards, name as 1.png, 2.png...",
            "* Higher priority vehicles are selected first",
            "* Start from garage screen"
        ]
        for tip in tips_text:
            ttk.Label(tips_frame, text=tip, anchor="w").pack(anchor="w", padx=5, pady=2)

        # === Control Section ===
        control_frame = ttk.LabelFrame(main_frame, text="Control", padding="5")
        control_frame.pack(fill=tk.X, pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Start", width=15, command=self._onStart).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", width=15, command=self._onStop).pack(side=tk.LEFT, padx=5)

        self._status_label = ttk.Label(control_frame, text="Status: Stopped", foreground="red")
        self._status_label.pack(pady=5)

        # === Duration & End Behavior Section ===
        duration_frame = ttk.LabelFrame(main_frame, text="Duration & End Behavior (Not Implemented)", padding="5")
        duration_frame.pack(fill=tk.X, pady=5)

        hours_frame = ttk.Frame(duration_frame)
        hours_frame.pack(fill=tk.X, pady=2)

        ttk.Label(hours_frame, text="Run duration limit:").pack(side=tk.LEFT, padx=5)
        hours_var = tk.IntVar(value=self._run_hours)
        hours_spinbox = ttk.Spinbox(hours_frame, from_=1, to=24, width=10, textvariable=hours_var,
                                    command=lambda: setattr(self, '_run_hours', hours_var.get()))
        hours_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(hours_frame, text="hours").pack(side=tk.LEFT, padx=5)

        auto_stop_var = tk.BooleanVar(value=self._auto_stop)
        ttk.Checkbutton(duration_frame, text="Auto stop when duration reached",
                       variable=auto_stop_var,
                       command=lambda: setattr(self, '_auto_stop', auto_stop_var.get())).pack(anchor="w", padx=5, pady=2)

        auto_shutdown_var = tk.BooleanVar(value=self._auto_shutdown)
        ttk.Checkbutton(duration_frame, text="Auto shutdown when duration reached (Admin required)",
                       variable=auto_shutdown_var,
                       command=lambda: setattr(self, '_auto_shutdown', auto_shutdown_var.get())).pack(anchor="w", padx=5, pady=2)

        self._end_time_label = ttk.Label(duration_frame, text="Estimated end time: --")
        self._end_time_label.pack(pady=2)

        # === Feature Expansion Section ===
        feature_frame = ttk.LabelFrame(main_frame, text="Feature Expansion (Not Implemented)", padding="5")
        feature_frame.pack(fill=tk.X, pady=5)

        silver_reserve_var = tk.BooleanVar(value=self._silver_reserve)
        ttk.Checkbutton(feature_frame, text="Enable silver reserve on start",
                       variable=silver_reserve_var,
                       command=lambda: setattr(self, '_silver_reserve', silver_reserve_var.get())).pack(anchor="w", padx=5, pady=2)

        # === Vehicle Priority Section ===
        vehicle_frame = ttk.LabelFrame(main_frame, text="Vehicle Priority", padding="5")
        vehicle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Label(vehicle_frame, text="Select vehicles in priority order:").pack(anchor="w", padx=5)
        ttk.Label(vehicle_frame, text="1.png -> 2.png -> 3.png ...", foreground="gray").pack(anchor="w", padx=5)

        button_frame2 = ttk.Frame(vehicle_frame)
        button_frame2.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame2, text="Add Screenshot", command=self._onAddScreenshot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame2, text="Open Screenshot Folder", command=self._onOpenScreenshotDir).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame2, text="Refresh", command=self._refreshVehicleList).pack(side=tk.LEFT, padx=5)

        ttk.Label(vehicle_frame, text="Current vehicle priority:").pack(anchor="w", padx=5, pady=5)

        # Scrollable vehicle list
        list_container = ttk.Frame(vehicle_frame)
        list_container.pack(fill=tk.BOTH, expand=True, pady=5)

        self._vehicle_canvas = tk.Canvas(list_container, height=200, borderwidth=1, relief=tk.SUNKEN)
        self._vehicle_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self._vehicle_canvas.yview)
        self._vehicle_frame = ttk.Frame(self._vehicle_canvas)

        self._vehicle_canvas.configure(yscrollcommand=self._vehicle_scrollbar.set)
        self._vehicle_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._vehicle_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._vehicle_canvas.create_window((0, 0), window=self._vehicle_frame, anchor="nw")
        self._vehicle_frame.bind("<Configure>", lambda e: self._vehicle_canvas.configure(scrollregion=self._vehicle_canvas.bbox("all")))

        # === Footer ===
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=5)

        ttk.Label(footer_frame, text="Version: v0.1.0", foreground="gray").pack(side=tk.LEFT, padx=5)
        ttk.Button(footer_frame, text="Open Log Folder", command=self._onOpenLogDir).pack(side=tk.RIGHT, padx=5)
        ttk.Button(footer_frame, text="Open Config", command=self._onOpenConfigDir).pack(side=tk.RIGHT, padx=5)

    # -------------------------------------------------------------------------
    # Vehicle List Management
    # -------------------------------------------------------------------------

    def _refreshVehicleList(self) -> None:
        """Refresh the vehicle thumbnail list."""
        if self._vehicle_frame is None:
            return

        # Clear existing widgets
        for widget in self._vehicle_frame.winfo_children():
            widget.destroy()

        # Clear old images
        self._vehicle_images.clear()

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

                # Convert BGR to RGB for PIL
                thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(thumbnail_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                self._vehicle_images[img_path.name] = photo

                # Create item frame
                item_frame = ttk.Frame(self._vehicle_frame)
                item_frame.pack(fill=tk.X, padx=5, pady=2)

                # Image label
                img_label = ttk.Label(item_frame, image=photo)
                img_label.pack(side=tk.LEFT, padx=5)

                # Filename label
                ttk.Label(item_frame, text=img_path.name).pack(side=tk.LEFT, padx=5)

                # Delete button
                ttk.Button(item_frame, text="Delete", width=10,
                          command=lambda p=img_path: self._onDeleteScreenshot(p)).pack(side=tk.RIGHT, padx=5)

            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")

        # Update scroll region
        self._vehicle_frame.update_idletasks()
        self._vehicle_canvas.configure(scrollregion=self._vehicle_canvas.bbox("all"))

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def _onStart(self) -> None:
        """Start task manager."""
        if self._is_running:
            logger.warning("Task already running")
            return

        self._task_manager = TaskManager(
            run_hours=self._run_hours,
            auto_stop=self._auto_stop,
            auto_shutdown=self._auto_shutdown,
            enable_silver_reserve=self._silver_reserve,
        )

        self._is_running = True
        self._start_time = datetime.now()
        self._task_thread = threading.Thread(
            target=self._task_manager.run_forever, daemon=True
        )
        self._task_thread.start()

        logger.info("Task manager started")

    def _onStop(self) -> None:
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

    def _onAddScreenshot(self) -> None:
        """Open file dialog to add screenshot."""
        file_path = filedialog.askopenfilename(
            title="Select Vehicle Screenshot",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
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
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete {img_path.name}?",
            parent=self._root
        )
        if result:
            try:
                if img_path.exists():
                    img_path.unlink()
                    self._refreshVehicleList()
                    logger.info(f"Deleted vehicle screenshot: {img_path}")
            except Exception as e:
                logger.error(f"Failed to delete screenshot: {e}")

    def _onOpenScreenshotDir(self) -> None:
        """Open vehicle screenshot directory."""
        try:
            if sys.platform == "win32":
                os.startfile(str(self._vehicle_screenshot_dir))
        except Exception as e:
            logger.error(f"Failed to open directory: {e}")

    def _onOpenLogDir(self) -> None:
        """Open log directory."""
        try:
            if sys.platform == "win32":
                os.startfile(str(GetLogDir()))
        except Exception as e:
            logger.error(f"Failed to open log directory: {e}")

    def _onOpenConfigDir(self) -> None:
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
        """Update status display."""
        if self._status_label is None:
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

                self._status_label.config(text=status_text, foreground="green")

                if self._end_time_label:
                    self._end_time_label.config(text=end_text)
            else:
                self._status_label.config(text="Status: Running", foreground="green")
        else:
            self._status_label.config(text="Status: Stopped", foreground="red")

            if self._end_time_label:
                self._end_time_label.config(text="Estimated end time: --")

        # Schedule next update
        if self._root:
            self._root.after(1000, self._updateStatus)

    # -------------------------------------------------------------------------
    # Main Entry
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Create Tkinter window and start UI loop (blocking)."""
        # Build UI
        self._buildUI()

        # Load initial vehicle list
        self._refreshVehicleList()

        # Start status update loop
        self._updateStatus()

        # Start main loop
        self._root.mainloop()


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
