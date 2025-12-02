import sys
import os
import shutil
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from loguru import logger
import cv2

from src.core.task_manager import TaskManager
from src.utils.global_path import GetVehicleScreenshotsDir, GetLogDir, GetConfigPath

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TANK ROBOT")
        self.root.geometry("900x1440")
        self.root.configure(bg="#F0F0F0")

        # State
        self.is_running = False
        self.run_hours = tk.IntVar(value=4)
        self.auto_stop = tk.BooleanVar(value=False)
        self.auto_shutdown = tk.BooleanVar(value=False)
        self.silver_reserve = tk.BooleanVar(value=False)
        
        self.vehicle_images = []  # List of (path, thumbnail) tuples
        self.vehicle_screenshot_dir = self._init_vehicle_screenshot_dir()
        
        # TaskManager
        self.task_manager_ = None
        self.task_thread_ = None

        # InitLog
        self._init_log()

        # Setup UI
        self._build_ui()
        self._update_status()

    def _init_log(self):
        """Initialize logging configuration"""
        log_dir = GetLogDir()
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.remove()  # Remove default handler
        
        # Console handler
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # File handler
        log_file = log_dir / "wot_ai_{time:YYYY-MM-DD}.log"
        logger.add(
            str(log_file),
            rotation="00:00",  # New file every day at midnight
            retention="10 days",  # Keep logs for 10 days
            level="DEBUG",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
        
        logger.info(f"Log initialized, saving to: {log_dir}")

    def _create_section(self, parent, title=None):
        """Create a section frame with optional title"""
        frame = tk.Frame(parent, bg="#F0F0F0", relief=tk.RAISED, bd=1)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        if title:
            title_label = tk.Label(
                frame, text=title, font=("Microsoft YaHei", 10, "bold"),
                bg="#F0F0F0", anchor="w"
            )
            title_label.pack(fill=tk.X, padx=5, pady=2)
            tk.Frame(frame, height=1, bg="#CCCCCC").pack(fill=tk.X, padx=5)
        
        content = tk.Frame(frame, bg="#F0F0F0")
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        return content

    def _build_ui(self):
        # 使用提示区
        tips_section = self._create_section(self.root, "使用提示区")
        tips = [
            "· 请将游戏设置为\"窗口化全屏\"",
            "· 推荐画质：中–低",
            "· 游戏进入车库后按 F9 启动 / F10 停止",
        ]
        for tip in tips:
            tk.Label(
                tips_section, text=tip, font=("Microsoft YaHei", 9),
                bg="#F0F0F0", anchor="w"
            ).pack(anchor="w", padx=10, pady=2)

        # 核心控制区
        ctrl_section = self._create_section(self.root, "核心控制区")
        btn_frame = tk.Frame(ctrl_section, bg="#F0F0F0")
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame, text="● 启动", command=self._start,
            font=("Microsoft YaHei", 10), width=15
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame, text="■ 停止", command=self._stop,
            font=("Microsoft YaHei", 10), width=15
        ).pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(
            ctrl_section, text="状态：● 已停止",
            font=("Microsoft YaHei", 10), bg="#F0F0F0"
        )
        self.status_label.pack(pady=5)

        # 时长 & 结束行为配置区
        time_section = self._create_section(self.root, "时长 & 结束行为配置区")
        
        time_row = tk.Frame(time_section, bg="#F0F0F0")
        time_row.pack(anchor="w", pady=5)
        tk.Label(
            time_row, text="运行时长限制：", font=("Microsoft YaHei", 9),
            bg="#F0F0F0"
        ).pack(side=tk.LEFT, padx=5)
        tk.Entry(
            time_row, textvariable=self.run_hours, width=6,
            font=("Microsoft YaHei", 9)
        ).pack(side=tk.LEFT, padx=5)
        tk.Label(
            time_row, text="小时", font=("Microsoft YaHei", 9),
            bg="#F0F0F0"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Checkbutton(
            time_section, text="达到时长后自动停止",
            variable=self.auto_stop, font=("Microsoft YaHei", 9),
            bg="#F0F0F0", anchor="w"
        ).pack(anchor="w", padx=5, pady=2)
        
        tk.Checkbutton(
            time_section, text="达到时长后自动关机（需要管理员权限）",
            variable=self.auto_shutdown, font=("Microsoft YaHei", 9),
            bg="#F0F0F0", anchor="w"
        ).pack(anchor="w", padx=5, pady=2)
        
        self.end_time_label = tk.Label(
            time_section, text="预计结束时间：--",
            font=("Microsoft YaHei", 9), bg="#F0F0F0", anchor="w"
        )
        self.end_time_label.pack(anchor="w", padx=5, pady=5)

        # 功能扩展区
        feature_section = self._create_section(self.root, "功能扩展区")
        
        tk.Checkbutton(
            feature_section, text="是否启用启动银币储备",
            variable=self.silver_reserve, font=("Microsoft YaHei", 9),
            bg="#F0F0F0", anchor="w"
        ).pack(anchor="w", padx=5, pady=2)

        # 车辆优先级设置区
        vehicle_section = self._create_section(self.root, "车辆优先级设置区")
        desc_text = (
            "描述：按优先级顺序选择可出战车辆\n"
            " 1.png → 2.png → 3.png …\n"
            "（若当前车辆模板检测不到，则顺延至下一辆车）"
        )
        tk.Label(
            vehicle_section, text=desc_text, font=("Microsoft YaHei", 9),
            bg="#F0F0F0", anchor="w", justify=tk.LEFT
        ).pack(anchor="w", padx=5, pady=5)
        
        btn_row = tk.Frame(vehicle_section, bg="#F0F0F0")
        btn_row.pack(pady=5)
        tk.Button(
            btn_row, text="添加车辆截图", command=self._add_vehicle_screenshot,
            font=("Microsoft YaHei", 9), width=15
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_row, text="打开车辆截图目录", command=self._open_screenshot_dir,
            font=("Microsoft YaHei", 9), width=15
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            vehicle_section, text="当前车辆优先级",
            font=("Microsoft YaHei", 9), bg="#F0F0F0", anchor="w"
        ).pack(anchor="w", padx=5, pady=5)
        
        # Vehicle list frame with scrollbar
        list_frame = tk.Frame(vehicle_section, bg="#F0F0F0", relief=tk.SUNKEN, bd=1)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(list_frame, bg="white", height=150)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.vehicle_list_frame = tk.Frame(canvas, bg="white")
        
        canvas.create_window((0, 0), window=self.vehicle_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.vehicle_canvas = canvas
        self._refresh_vehicle_list()

        # 底部栏
        footer = tk.Frame(self.root, bg="#F0F0F0")
        footer.pack(fill=tk.X, pady=10)
        
        tk.Label(
            footer, text="版本：v0.3.1", font=("Microsoft YaHei", 9),
            bg="#F0F0F0"
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            footer, text="查看日志", command=self._view_logs,
            font=("Microsoft YaHei", 9), relief=tk.FLAT
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            footer, text="配置文件目录", command=self._open_config_dir,
            font=("Microsoft YaHei", 9), relief=tk.FLAT
        ).pack(side=tk.LEFT, padx=5)

        # 单步调试区
        debug_section = self._create_section(self.root, "单步调试区")
        self._build_debug_ui(debug_section)

    def _start(self):
        if self.is_running:
            logger.warning("任务已在运行")
            return

        # 创建TaskManager
        self.task_manager_ = TaskManager(
            run_hours=int(self.run_hours.get()),
            auto_stop=self.auto_stop.get(),
            auto_shutdown=self.auto_shutdown.get(),
        )
        
        # 在独立线程中运行TaskManager
        self.is_running = True
        self.task_thread_ = threading.Thread(target=self.task_manager_.run_forever, daemon=True)
        self.task_thread_.start()
        
        logger.info("任务管理器已启动")
        self._update_status()

    def _stop(self):
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.task_manager_:
            self.task_manager_.stop()
        
        if self.task_thread_ and self.task_thread_.is_alive():
            self.task_thread_.join(timeout=2.0)
        
        self.task_manager_ = None
        logger.info("任务管理器已停止")
        self._update_status()
    
    def _init_vehicle_screenshot_dir(self) -> Path:
        """根据配置初始化车辆截图目录"""
        default_dir = GetVehicleScreenshotsDir()
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir

    def _update_status(self):
        if self.is_running:
            try:
                hrs = int(self.run_hours.get())
                if hrs > 0:
                    end = datetime.now() + timedelta(hours=hrs)
                    remaining = end - datetime.now()
                    hours = int(remaining.total_seconds() // 3600)
                    minutes = int((remaining.total_seconds() % 3600) // 60)
                    seconds = int(remaining.total_seconds() % 60)
                    status_text = f"状态：○ 运行中 (剩余 {hours:02d}:{minutes:02d}:{seconds:02d})"
                    self.status_label.config(text=status_text)
                    self.end_time_label.config(text=f"预计结束时间：{end:%Y-%m-%d %H:%M}")
            except:
                status_text = "状态：○ 运行中"
                self.status_label.config(text=status_text)
        else:
            self.status_label.config(text="状态：● 已停止")
            self.end_time_label.config(text="预计结束时间：--")
        
        self.root.after(3000, self._update_status)

    def _add_vehicle_screenshot(self):
        file_path = filedialog.askopenfilename(
            title="选择车辆截图",
            filetypes=[("图片文件", "*.png"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                dest_path = self.vehicle_screenshot_dir / Path(file_path).name
                shutil.copy2(file_path, dest_path)
                self._refresh_vehicle_list()
                logger.info(f"已添加车辆截图: {dest_path}")
            except Exception as e:
                messagebox.showerror("错误", f"添加截图失败: {e}")
                logger.error(f"添加截图失败: {e}")

    def _open_screenshot_dir(self):
        try:
            if sys.platform == "win32":
                os.startfile(str(self.vehicle_screenshot_dir))
        except Exception as e:
            messagebox.showerror("错误", f"打开目录失败: {e}")
            logger.error(f"打开目录失败: {e}")

    def _refresh_vehicle_list(self):
        """Refresh the vehicle list display"""
        # Clear existing widgets
        for widget in self.vehicle_list_frame.winfo_children():
            widget.destroy()
        
        # Load images from directory
        self.vehicle_images = []
        if self.vehicle_screenshot_dir.exists():
            image_files = sorted(self.vehicle_screenshot_dir.glob("*.png"))
            
            for idx, img_path in enumerate(image_files):
                try:
                    # Create thumbnail using OpenCV and save to temp file
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        scale = min(80/w, 60/h, 1.0)
                        new_w, new_h = int(w*scale), int(h*scale)
                        thumbnail = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                        # Save thumbnail to temp file
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        cv2.imwrite(temp_file.name, thumbnail)
                        temp_file.close()
                        
                        # Load as PhotoImage
                        photo = tk.PhotoImage(file=temp_file.name)
                        
                        # Create item frame
                        item_frame = tk.Frame(self.vehicle_list_frame, bg="white", relief=tk.RAISED, bd=1)
                        item_frame.pack(side=tk.LEFT, padx=5, pady=5)
                        
                        # Thumbnail
                        thumb_label = tk.Label(item_frame, image=photo, bg="white")
                        thumb_label.image = photo  # Keep a reference
                        thumb_label.temp_file = temp_file.name  # Keep temp file path
                        thumb_label.pack(padx=2, pady=2)
                        
                        # Filename
                        tk.Label(
                            item_frame, text=img_path.name, font=("Microsoft YaHei", 8),
                            bg="white"
                        ).pack()
                        
                        # Delete button
                        del_btn = tk.Button(
                            item_frame, text="删除 ❌", font=("Microsoft YaHei", 7),
                            command=lambda p=img_path: self._delete_vehicle_screenshot(p),
                            bg="#FFCCCC", width=10
                        )
                        del_btn.pack(pady=2)
                        
                        self.vehicle_images.append((img_path, photo))
                except Exception as e:
                    logger.error(f"加载图片失败 {img_path}: {e}")
        
        # Update scroll region
        self.vehicle_list_frame.update_idletasks()
        self.vehicle_canvas.configure(scrollregion=self.vehicle_canvas.bbox("all"))

    def _delete_vehicle_screenshot(self, img_path):
        try:
            if messagebox.askyesno("确认", f"确定要删除 {img_path.name} 吗？"):
                img_path.unlink()
                self._refresh_vehicle_list()
                logger.info(f"已删除车辆截图: {img_path}")
        except Exception as e:
            messagebox.showerror("错误", f"删除失败: {e}")
            logger.error(f"删除失败: {e}")

    def _view_logs(self):
        os.startfile(str(GetLogDir()))

    def _open_config_dir(self):
        os.startfile(str(GetConfigPath()))

    def run(self):
        self.root.mainloop()


def main():
    try:
        window = MainWindow()
        window.run()
        return 0
    except Exception as e:
        logger.error(f"主程序运行错误: {e}")
        import traceback
        traceback.print_exc()
        return -1

if __name__ == "__main__":
    main()