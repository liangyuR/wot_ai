#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main control window (Tkinter) with Chinese UI."""

import os
import shutil
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from loguru import logger

from src.core.task_manager import TaskManager
from src.utils.global_path import GetConfigPath, GetGlobalConfig, GetLogDir, GetVehicleScreenshotsDir


class MainWindow:
    """Main control window using Tkinter."""

    def __init__(self):
        self._title = "坦克助手控制台"
        self._window_size = (560, 1080)

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
            rotation="00:00",
            retention="10 days",
            level="DEBUG",
            filter=lambda record: record["level"].name == "DEBUG",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )

        # INFO file handler - INFO and above
        info_log_file = log_dir / "wot_ai_info_{time:YYYY-MM-DD}.log"
        logger.add(
            str(info_log_file),
            rotation="00:00",
            retention="10 days",
            level="INFO",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
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

    def _initStyle(self) -> None:
        """Initialize ttk style for a modern dark look."""
        style = ttk.Style(self._root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        bg = "#0f1115"
        card = "#161920"
        accent = "#4f9cff"
        text = "#e5e7eb"
        subtle = "#9ca3af"

        style.configure(".", background=bg, foreground=text, font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Body.TLabel", background=bg, foreground=text, font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background=bg, foreground=text, font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("SubTitle.TLabel", background=bg, foreground=subtle, font=("Microsoft YaHei UI", 10))
        style.configure("Card.TFrame", background=card, relief="flat", borderwidth=0)
        style.configure("CardTitle.TLabel", background=card, foreground=text, font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("StatusStopped.TLabel", background=card, foreground="#ff6b6b", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("StatusRunning.TLabel", background=card, foreground="#1dd1a1", font=("Microsoft YaHei UI", 12, "bold"))

        style.configure("TButton", padding=8, relief="flat", background=card, foreground=text, font=("Microsoft YaHei UI", 10))
        style.map("TButton", background=[("active", "#1f2937")])
        style.configure("Accent.TButton", background=accent, foreground="#0b1220", font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#78b5ff")], foreground=[("active", "#0b1220")])

        self._root.option_add("*TCombobox*Listbox*Font", ("Microsoft YaHei UI", 10))

    def _buildUI(self) -> None:
        """Build the main UI layout."""
        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.geometry(f"{self._window_size[0]}x{self._window_size[1]}")
        self._root.resizable(False, False)
        self._root.configure(bg="#0f1115")
        self._initStyle()

        # Main container with padding
        main_frame = ttk.Frame(self._root, padding="12")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(main_frame, padding="8")
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="坦克世界 · 自动导航助手", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="启动导航、管理载具优先级、查看运行状态", style="SubTitle.TLabel").pack(anchor="w", pady=(2, 0))

        # Tips
        tips_frame = ttk.LabelFrame(main_frame, text="快速提示", padding="10")
        tips_frame.pack(fill=tk.X, pady=6)
        tips_text = [
            "1) 游戏窗口化 1920x1080，画质中低。",
            "2) 车库界面截图载具卡，命名 1.png/2.png/3.png 表示优先级。",
            "3) 先编辑 config/config.yaml，再启动导航。",
            "4) 建议保持日志可见，方便排查。",
        ]
        for tip in tips_text:
            ttk.Label(tips_frame, text=tip, anchor="w", style="Body.TLabel").pack(anchor="w", padx=4, pady=1)

        # Control
        control_frame = ttk.LabelFrame(main_frame, text="控制中心", padding="10")
        control_frame.pack(fill=tk.X, pady=6)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=6)

        ttk.Button(button_frame, text="▶ 启动导航", width=16, style="Accent.TButton", command=self._onStart).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="■ 停止", width=12, command=self._onStop).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="⚙ 打开配置", width=12, command=self._onOpenConfigDir).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="🧾 查看日志", width=12, command=self._onOpenLogDir).pack(side=tk.RIGHT, padx=5)

        status_card = ttk.Frame(control_frame, padding="10", style="Card.TFrame")
        status_card.pack(fill=tk.X, pady=4)
        ttk.Label(status_card, text="运行状态", style="CardTitle.TLabel").pack(anchor="w")
        self._status_label = ttk.Label(status_card, text="未运行", style="StatusStopped.TLabel")
        self._status_label.pack(anchor="w", pady=(4, 2))

        # Duration
        duration_frame = ttk.LabelFrame(main_frame, text="运行时长与结束动作（预留）", padding="10")
        duration_frame.pack(fill=tk.X, pady=6)

        hours_frame = ttk.Frame(duration_frame)
        hours_frame.pack(fill=tk.X, pady=2)
        ttk.Label(hours_frame, text="限制运行时长：").pack(side=tk.LEFT, padx=5)
        hours_var = tk.IntVar(value=self._run_hours)
        hours_spinbox = ttk.Spinbox(hours_frame, from_=1, to=24, width=10, textvariable=hours_var,
                                    command=lambda: setattr(self, '_run_hours', hours_var.get()))
        hours_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(hours_frame, text="小时").pack(side=tk.LEFT, padx=5)

        auto_stop_var = tk.BooleanVar(value=self._auto_stop)
        ttk.Checkbutton(duration_frame, text="到达时长后自动停止",
                        variable=auto_stop_var,
                        command=lambda: setattr(self, '_auto_stop', auto_stop_var.get())).pack(anchor="w", padx=5, pady=2)

        auto_shutdown_var = tk.BooleanVar(value=self._auto_shutdown)
        ttk.Checkbutton(duration_frame, text="到达时长后自动关机（需管理员）",
                        variable=auto_shutdown_var,
                        command=lambda: setattr(self, '_auto_shutdown', auto_shutdown_var.get())).pack(anchor="w", padx=5, pady=2)

        self._end_time_label = ttk.Label(duration_frame, text="预计结束时间：--")
        self._end_time_label.pack(pady=2)

        # Feature (placeholder)
        feature_frame = ttk.LabelFrame(main_frame, text="功能扩展（预留）", padding="10")
        feature_frame.pack(fill=tk.X, pady=6)

        silver_reserve_var = tk.BooleanVar(value=self._silver_reserve)
        ttk.Checkbutton(feature_frame, text="启动时开启银币储备（预留）",
                        variable=silver_reserve_var,
                        command=lambda: setattr(self, '_silver_reserve', silver_reserve_var.get())).pack(anchor="w", padx=5, pady=2)

        # Vehicle Priority
        vehicle_frame = ttk.LabelFrame(main_frame, text="载具优先级", padding="10")
        vehicle_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        ttk.Label(vehicle_frame, text="按优先级选择载具：1.png → 2.png → 3.png ...").pack(anchor="w", padx=5)
        ttk.Label(vehicle_frame, text="高优先级先被选择；图片放在 vehicle_screenshots 目录。", foreground="gray").pack(anchor="w", padx=5, pady=(0, 4))

        button_frame2 = ttk.Frame(vehicle_frame)
        button_frame2.pack(fill=tk.X, pady=6)

        ttk.Button(button_frame2, text="➕ 添加截图", command=self._onAddScreenshot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame2, text="📂 打开截图目录", command=self._onOpenScreenshotDir).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame2, text="↻ 刷新", command=self._refreshVehicleList).pack(side=tk.LEFT, padx=5)

        ttk.Label(vehicle_frame, text="当前载具顺序：").pack(anchor="w", padx=5, pady=4)

        list_container = ttk.Frame(vehicle_frame)
        list_container.pack(fill=tk.BOTH, expand=True, pady=4)

        self._vehicle_canvas = tk.Canvas(list_container, height=240, borderwidth=0, highlightthickness=0, bg="#0f1115")
        self._vehicle_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self._vehicle_canvas.yview)
        self._vehicle_frame = ttk.Frame(self._vehicle_canvas)

        self._vehicle_canvas.configure(yscrollcommand=self._vehicle_scrollbar.set)
        self._vehicle_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._vehicle_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._vehicle_canvas.create_window((0, 0), window=self._vehicle_frame, anchor="nw")
        self._vehicle_frame.bind("<Configure>", lambda e: self._vehicle_canvas.configure(scrollregion=self._vehicle_canvas.bbox("all")))

        # Footer
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=6)
        ttk.Label(footer_frame, text="版本 v0.1.0", foreground="gray").pack(side=tk.LEFT, padx=5)
        ttk.Label(footer_frame, text="配置路径：config/config.yaml", foreground="gray").pack(side=tk.RIGHT, padx=5)

    # -------------------------------------------------------------------------
    # Vehicle List Management
    # -------------------------------------------------------------------------

    def _refreshVehicleList(self) -> None:
        """Refresh the vehicle thumbnail list."""
        if self._vehicle_frame is None:
            return

        for widget in self._vehicle_frame.winfo_children():
            widget.destroy()

        self._vehicle_images.clear()

        if not self._vehicle_screenshot_dir.exists():
            return

        image_files = sorted(self._vehicle_screenshot_dir.glob("*.png"))

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                max_thumb_size = 80
                scale = min(max_thumb_size / w, max_thumb_size / h, 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                thumbnail = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(thumbnail_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                self._vehicle_images[img_path.name] = photo

                item_frame = ttk.Frame(self._vehicle_frame, padding="4")
                item_frame.pack(fill=tk.X, padx=5, pady=3)

                ttk.Label(item_frame, image=photo).pack(side=tk.LEFT, padx=6)
                ttk.Label(item_frame, text=img_path.name).pack(side=tk.LEFT, padx=6)

                ttk.Button(item_frame, text="删除", width=8,
                           command=lambda p=img_path: self._onDeleteScreenshot(p)).pack(side=tk.RIGHT, padx=6)

            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")

        self._vehicle_frame.update_idletasks()
        self._vehicle_canvas.configure(scrollregion=self._vehicle_canvas.bbox("all"))

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def _onStart(self) -> None:
        """Start task manager."""
        if self._is_running:
            logger.warning("任务已在运行")
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

        logger.info("任务管理器已启动")

        # 将焦点切换到游戏窗口
        self._focusGameWindow()

    def _focusGameWindow(self) -> None:
        """将焦点切换到游戏窗口"""
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            # 获取游戏进程名
            config = GetGlobalConfig()
            process_name = getattr(config.game, "process_name", "WorldOfTanks.exe")

            # EnumWindows 回调
            EnumWindowsProc = ctypes.WINFUNCTYPE(
                wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
            )

            target_hwnd = None

            def enum_callback(hwnd, lparam):
                nonlocal target_hwnd
                if not user32.IsWindowVisible(hwnd):
                    return True

                # 获取窗口进程 ID
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

                # 获取进程名
                try:
                    h_process = kernel32.OpenProcess(0x0410, False, pid.value)  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
                    if h_process:
                        exe_path = ctypes.create_unicode_buffer(260)
                        size = wintypes.DWORD(260)
                        if kernel32.QueryFullProcessImageNameW(h_process, 0, exe_path, ctypes.byref(size)):
                            if process_name.lower() in exe_path.value.lower():
                                target_hwnd = hwnd
                                kernel32.CloseHandle(h_process)
                                return False  # 停止枚举
                        kernel32.CloseHandle(h_process)
                except Exception:
                    pass
                return True

            user32.EnumWindows(EnumWindowsProc(enum_callback), 0)

            if target_hwnd:
                # 激活窗口
                user32.SetForegroundWindow(target_hwnd)
                logger.info(f"已将焦点切换到游戏窗口 (hwnd={target_hwnd})")
            else:
                logger.warning(f"未找到游戏窗口: {process_name}")

        except Exception as e:
            logger.error(f"切换游戏窗口焦点失败: {e}")

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
        logger.info("任务管理器已停止")

    def _onAddScreenshot(self) -> None:
        """Open file dialog to add screenshot."""
        file_path = filedialog.askopenfilename(
            title="选择载具截图",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            try:
                src_path = Path(file_path)
                dest_path = self._vehicle_screenshot_dir / src_path.name
                shutil.copy2(src_path, dest_path)
                self._refreshVehicleList()
                logger.info(f"添加载具截图: {dest_path}")
            except Exception as e:
                logger.error(f"添加截图失败: {e}")

    def _onDeleteScreenshot(self, img_path: Path) -> None:
        """Show delete confirmation dialog."""
        result = messagebox.askyesno(
            "确认删除",
            f"确定删除 {img_path.name} ？",
            parent=self._root
        )
        if result:
            try:
                if img_path.exists():
                    img_path.unlink()
                    self._refreshVehicleList()
                    logger.info(f"已删除载具截图: {img_path}")
            except Exception as e:
                logger.error(f"删除截图失败: {e}")

    def _onOpenScreenshotDir(self) -> None:
        """Open vehicle screenshot directory."""
        try:
            if sys.platform == "win32":
                os.startfile(str(self._vehicle_screenshot_dir))
        except Exception as e:
            logger.error(f"打开目录失败: {e}")

    def _onOpenLogDir(self) -> None:
        """Open log directory."""
        try:
            if sys.platform == "win32":
                os.startfile(str(GetLogDir()))
        except Exception as e:
            logger.error(f"打开日志目录失败: {e}")

    def _onOpenConfigDir(self) -> None:
        """Open config file location."""
        try:
            if sys.platform == "win32":
                os.startfile(str(GetConfigPath()))
        except Exception as e:
            logger.error(f"打开配置失败: {e}")

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
                    status_text = f"运行中 | 剩余 {hours:02d}:{minutes:02d}:{seconds:02d}"
                    end_text = f"预计结束时间：{end_time:%Y-%m-%d %H:%M}"
                else:
                    status_text = "运行中 | 已超过预设时间"
                    end_text = "预计结束时间：--"

                self._status_label.config(text=status_text, style="StatusRunning.TLabel")

                if self._end_time_label:
                    self._end_time_label.config(text=end_text)
            else:
                self._status_label.config(text="运行中", style="StatusRunning.TLabel")
        else:
            self._status_label.config(text="未运行", style="StatusStopped.TLabel")
            if self._end_time_label:
                self._end_time_label.config(text="预计结束时间：--")

        if self._root:
            self._root.after(1000, self._updateStatus)

    # -------------------------------------------------------------------------
    # Main Entry
    # -------------------------------------------------------------------------

    def run(self, auto_start: bool = False) -> None:
        """Create Tkinter window and start UI loop (blocking).

        Args:
            auto_start: 是否在启动后自动开始任务（用于 CUDA 错误重启后恢复）
        """
        self._buildUI()
        self._refreshVehicleList()
        self._updateStatus()

        # 自动启动（延迟执行，确保 UI 完全初始化）
        if auto_start:
            logger.info("检测到 --auto-start 参数，将在 2 秒后自动启动...")
            self._root.after(2000, self._onStart)

        # Start main loop
        self._root.mainloop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="WoT AI Tank Robot")
    parser.add_argument("--auto-start", action="store_true", 
                       help="自动启动任务（用于 CUDA 错误重启后恢复）")
    args = parser.parse_args()

    try:
        window = MainWindow()
        window.run(auto_start=args.auto_start)
        return 0
    except Exception as e:
        logger.error(f"Main program error: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
