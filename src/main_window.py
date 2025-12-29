#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main control window (Tkinter) with Chinese UI."""

import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from loguru import logger

from src.core.battle_task import BattleTask
from src.utils.global_path import GetConfigPath, GetGlobalConfig, GetLogDir, GetVehicleScreenshotsDir


class MainWindow:
    """Main control window using Tkinter."""

    # UI Constants
    WINDOW_WIDTH = 560
    WINDOW_HEIGHT = 1080
    COLOR_BG = "#0f1115"
    COLOR_CARD = "#161920"
    COLOR_ACCENT = "#4f9cff"
    COLOR_TEXT = "#e5e7eb"
    COLOR_SUBTLE = "#9ca3af"
    COLOR_STATUS_STOPPED = "#ff6b6b"
    COLOR_STATUS_RUNNING = "#1dd1a1"
    COLOR_BUTTON_ACTIVE = "#1f2937"
    COLOR_ACCENT_ACTIVE = "#78b5ff"
    COLOR_ACCENT_FG = "#0b1220"

    FONT_FAMILY = "Microsoft YaHei UI"
    FONT_SIZE_NORMAL = 10
    FONT_SIZE_TITLE = 16
    FONT_SIZE_SUBTITLE = 10
    FONT_SIZE_CARD_TITLE = 11
    FONT_SIZE_STATUS = 12

    # Business Constants
    MAX_THUMBNAIL_SIZE = 80
    AUTO_START_DELAY_MS = 2000
    STATUS_UPDATE_INTERVAL_MS = 1000
    DEFAULT_RUN_HOURS = 4

    # Windows API Constants
    PROCESS_QUERY_INFORMATION = 0x0410
    MAX_PATH_LENGTH = 260

    def __init__(self) -> None:
        """Initialize MainWindow."""
        self._title: str = "坦克助手控制台"
        self._window_size: Tuple[int, int] = (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        # State
        self._is_running: bool = False
        self._run_hours: int = self.DEFAULT_RUN_HOURS
        self._auto_stop: bool = False
        self._auto_shutdown: bool = False
        self._start_time: Optional[datetime] = None

        # UI Variables (Tkinter variables)
        self._auto_stop_var: Optional[tk.BooleanVar] = None
        self._auto_shutdown_var: Optional[tk.BooleanVar] = None
        self._silver_reserve: Optional[tk.BooleanVar] = None

        # Vehicle images: {filename: PhotoImage}
        self._vehicle_images: Dict[str, ImageTk.PhotoImage] = {}
        self._vehicle_screenshot_dir: Path = self._initVehicleScreenshotDir()

        # BattleTask
        self._battle_task: Optional[BattleTask] = None

        # Tkinter widgets
        self._root: Optional[tk.Tk] = None
        self._status_label: Optional[ttk.Label] = None
        self._end_time_label: Optional[ttk.Label] = None
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

        bg = self.COLOR_BG
        card = self.COLOR_CARD
        accent = self.COLOR_ACCENT
        text = self.COLOR_TEXT
        subtle = self.COLOR_SUBTLE

        style.configure(".", background=bg, foreground=text, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL))
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Body.TLabel", background=bg, foreground=text, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL))
        style.configure("Title.TLabel", background=bg, foreground=text, font=(self.FONT_FAMILY, self.FONT_SIZE_TITLE, "bold"))
        style.configure("SubTitle.TLabel", background=bg, foreground=subtle, font=(self.FONT_FAMILY, self.FONT_SIZE_SUBTITLE))
        style.configure("Card.TFrame", background=card, relief="flat", borderwidth=0)
        style.configure("CardTitle.TLabel", background=card, foreground=text, font=(self.FONT_FAMILY, self.FONT_SIZE_CARD_TITLE, "bold"))
        style.configure("StatusStopped.TLabel", background=card, foreground=self.COLOR_STATUS_STOPPED, font=(self.FONT_FAMILY, self.FONT_SIZE_STATUS, "bold"))
        style.configure("StatusRunning.TLabel", background=card, foreground=self.COLOR_STATUS_RUNNING, font=(self.FONT_FAMILY, self.FONT_SIZE_STATUS, "bold"))

        style.configure("TButton", padding=8, relief="flat", background=card, foreground=text, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL))
        style.map("TButton", background=[("active", self.COLOR_BUTTON_ACTIVE)])
        style.configure("Accent.TButton", background=accent, foreground=self.COLOR_ACCENT_FG, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))
        style.map("Accent.TButton", background=[("active", self.COLOR_ACCENT_ACTIVE)], foreground=[("active", self.COLOR_ACCENT_FG)])

        self._root.option_add("*TCombobox*Listbox*Font", (self.FONT_FAMILY, self.FONT_SIZE_NORMAL))

    def _buildUI(self) -> None:
        """Build the main UI layout."""
        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.geometry(f"{self._window_size[0]}x{self._window_size[1]}")
        self._root.resizable(False, False)
        self._root.configure(bg=self.COLOR_BG)
        self._initStyle()

        # Main container with padding
        main_frame = ttk.Frame(self._root, padding="12")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Build UI components
        self._buildHeader(main_frame)
        self._buildTips(main_frame)
        self._buildControlPanel(main_frame)
        self._buildDurationSettings(main_frame)
        self._buildFeatureSettings(main_frame)
        self._buildVehiclePriority(main_frame)
        self._buildFooter(main_frame)

    def _buildHeader(self, parent: ttk.Frame) -> None:
        """Build header section."""
        header = ttk.Frame(parent, padding="8")
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="坦克世界 · 自动导航助手", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="启动导航、管理载具优先级、查看运行状态", style="SubTitle.TLabel").pack(anchor="w", pady=(2, 0))

    def _buildTips(self, parent: ttk.Frame) -> None:
        """Build tips section."""
        tips_frame = ttk.LabelFrame(parent, text="快速提示", padding="10")
        tips_frame.pack(fill=tk.X, pady=6)
        tips_text = [
            "1) 游戏窗口化 1920x1080，画质中低。",
            "2) 车库界面截图载具卡，命名 1.png/2.png/3.png 表示优先级。",
            "3) 先编辑 config/config.yaml，再启动导航。",
            "4) 建议保持日志可见，方便排查。",
        ]
        for tip in tips_text:
            ttk.Label(tips_frame, text=tip, anchor="w", style="Body.TLabel").pack(anchor="w", padx=4, pady=1)

    def _buildControlPanel(self, parent: ttk.Frame) -> None:
        """Build control panel section."""
        control_frame = ttk.LabelFrame(parent, text="控制中心", padding="10")
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

    def _buildDurationSettings(self, parent: ttk.Frame) -> None:
        """Build duration settings section."""
        duration_frame = ttk.LabelFrame(parent, text="运行时长与结束动作（预留）", padding="10")
        duration_frame.pack(fill=tk.X, pady=6)

        hours_frame = ttk.Frame(duration_frame)
        hours_frame.pack(fill=tk.X, pady=2)
        ttk.Label(hours_frame, text="限制运行时长：").pack(side=tk.LEFT, padx=5)
        hours_var = tk.IntVar(value=self._run_hours)
        hours_spinbox = ttk.Spinbox(hours_frame, from_=1, to=24, width=10, textvariable=hours_var,
                                    command=lambda: setattr(self, '_run_hours', hours_var.get()))
        hours_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(hours_frame, text="小时").pack(side=tk.LEFT, padx=5)

        self._auto_stop_var = tk.BooleanVar(value=self._auto_stop)
        ttk.Checkbutton(duration_frame, text="到达时长后自动停止",
                        variable=self._auto_stop_var,
                        command=lambda: setattr(self, '_auto_stop', self._auto_stop_var.get())).pack(anchor="w", padx=5, pady=2)

        self._auto_shutdown_var = tk.BooleanVar(value=self._auto_shutdown)
        ttk.Checkbutton(duration_frame, text="到达时长后自动关机（需管理员）",
                        variable=self._auto_shutdown_var,
                        command=lambda: setattr(self, '_auto_shutdown', self._auto_shutdown_var.get())).pack(anchor="w", padx=5, pady=2)

        self._end_time_label = ttk.Label(duration_frame, text="预计结束时间：--")
        self._end_time_label.pack(pady=2)

    def _buildFeatureSettings(self, parent: ttk.Frame) -> None:
        """Build feature settings section."""
        feature_frame = ttk.LabelFrame(parent, text="功能扩展（预留）", padding="10")
        feature_frame.pack(fill=tk.X, pady=6)

        config = GetGlobalConfig()
        self._silver_reserve = tk.BooleanVar(value=config.game.enable_silver_reserve)
        ttk.Checkbutton(feature_frame, text="启动时开启银币储备",
                        variable=self._silver_reserve,
                        command=lambda: setattr(config.game, 'enable_silver_reserve', self._silver_reserve.get())).pack(anchor="w", padx=5, pady=2)

    def _buildVehiclePriority(self, parent: ttk.Frame) -> None:
        """Build vehicle priority section."""
        vehicle_frame = ttk.LabelFrame(parent, text="载具优先级", padding="10")
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

        self._vehicle_canvas = tk.Canvas(list_container, height=240, borderwidth=0, highlightthickness=0, bg=self.COLOR_BG)
        self._vehicle_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self._vehicle_canvas.yview)
        self._vehicle_frame = ttk.Frame(self._vehicle_canvas)

        self._vehicle_canvas.configure(yscrollcommand=self._vehicle_scrollbar.set)
        self._vehicle_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._vehicle_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._vehicle_canvas.create_window((0, 0), window=self._vehicle_frame, anchor="nw")
        self._vehicle_frame.bind("<Configure>", lambda e: self._vehicle_canvas.configure(scrollregion=self._vehicle_canvas.bbox("all")))

    def _buildFooter(self, parent: ttk.Frame) -> None:
        """Build footer section."""
        footer_frame = ttk.Frame(parent)
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
                scale = min(self.MAX_THUMBNAIL_SIZE / w, self.MAX_THUMBNAIL_SIZE / h, 1.0)
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
        """Start battle task."""
        if self._is_running:
            logger.warning("任务已在运行")
            return

        if self._silver_reserve is None:
            logger.error("银币储备变量未初始化")
            return

        self._battle_task = BattleTask(
            enable_silver_reserve=self._silver_reserve.get(),
            run_hours=self._run_hours,
            auto_stop=self._auto_stop,
            auto_shutdown=self._auto_shutdown,
        )

        self._is_running = True
        self._start_time = datetime.now()
        
        if not self._battle_task.start():
            logger.error("战斗任务启动失败")
            self._is_running = False
            self._battle_task = None
            return

        logger.info("战斗任务已启动")

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
                    h_process = kernel32.OpenProcess(self.PROCESS_QUERY_INFORMATION, False, pid.value)
                    if h_process:
                        exe_path = ctypes.create_unicode_buffer(self.MAX_PATH_LENGTH)
                        size = wintypes.DWORD(self.MAX_PATH_LENGTH)
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
        """Stop battle task."""
        if not self._is_running:
            return

        self._is_running = False

        if self._battle_task:
            self._battle_task.stop()

        self._battle_task = None
        self._start_time = None
        logger.info("战斗任务已停止")

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

    def _openDirectory(self, path: Path) -> None:
        """Open directory in file explorer (Windows only)."""
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
        except Exception as e:
            logger.error(f"打开目录失败: {e}")

    def _onOpenScreenshotDir(self) -> None:
        """Open vehicle screenshot directory."""
        self._openDirectory(self._vehicle_screenshot_dir)

    def _onOpenLogDir(self) -> None:
        """Open log directory."""
        self._openDirectory(GetLogDir())

    def _onOpenConfigDir(self) -> None:
        """Open config file location."""
        self._openDirectory(GetConfigPath())

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
            self._root.after(self.STATUS_UPDATE_INTERVAL_MS, self._updateStatus)

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
            self._root.after(self.AUTO_START_DELAY_MS, self._onStart)

        # Start main loop
        self._root.mainloop()


def main() -> int:
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
