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
from time import sleep

from src.core.battle_task import BattleTask
from src.core.task_manager import TaskManager
from src.core.state_machine import StateMachine, GameState
from src.core.global_context import GlobalContext
from src.core.tank_selector import TankSelector
from src.core.ai_controller import AIController
from src.core.actions import screenshot, screenshot_with_key_hold
from src.ui_control.actions import UIActions
from src.utils.global_path import GetVehicleScreenshotsDir, GetConfigPath, GetConfigTemplatePath, GetProgramDir
from src.navigation.config.loader import load_config
from src.navigation.config.models import NavigationConfig
from src.vision.detection.map_name_detector import MapNameDetector


class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TANK ROBOT")
        self.root.geometry("900x1080")
        self.root.configure(bg="#F0F0F0")

        # State
        self.is_running = False
        self.run_hours = tk.IntVar(value=4)
        self.auto_stop = tk.BooleanVar(value=False)
        self.auto_shutdown = tk.BooleanVar(value=False)
        self.silver_reserve = tk.BooleanVar(value=False)
        
        self.vehicle_images = []  # List of (path, thumbnail) tuples
        self.vehicle_screenshot_dir = self._init_vehicle_screenshot_dir()
        self.global_context_ = GlobalContext()
        
        # TaskManager
        self.task_manager_ = None
        self.task_thread_ = None

        # Debug components
        self.debug_state_machine_ = None
        self.debug_map_detector_ = None
        self.debug_tank_selector_ = None
        self.debug_ai_controller_ = None
        self.debug_battle_task = None
        self.debug_ai_config_ = None
        self.debug_ai_running_ = False

        # Setup UI
        self._build_ui()
        self._update_status()
        self._init_debug_components()

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
            btn_frame, text="● 启动 (F9)", command=self._start,
            font=("Microsoft YaHei", 10), width=15
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame, text="■ 停止 (F10)", command=self._stop,
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
        
        # 获取车辆优先级列表
        vehicle_priority = self._get_vehicle_priority()
        if not vehicle_priority:
            messagebox.showwarning("警告", "请先添加车辆截图")
            return
        
        # 准备AI配置
        ai_config = self._get_ai_config()
        
        # 创建TaskManager
        self.task_manager_ = TaskManager(
            vehicle_screenshot_dir=self.vehicle_screenshot_dir,
            vehicle_priority=vehicle_priority,
            ai_config=ai_config,
            run_hours=int(self.run_hours.get()),
            auto_stop=self.auto_stop.get(),
            auto_shutdown=self.auto_shutdown.get(),
            global_context=self.global_context_
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
    
    def _get_vehicle_priority(self) -> list:
        """获取车辆优先级列表"""
        if not self.vehicle_screenshot_dir.exists():
            return []
        
        image_files = sorted(self.vehicle_screenshot_dir.glob("*.png"))
        image_files.extend(sorted(self.vehicle_screenshot_dir.glob("*.jpg")))
        image_files.extend(sorted(self.vehicle_screenshot_dir.glob("*.jpeg")))
        
        return [f.name for f in image_files]
    
    def _init_vehicle_screenshot_dir(self) -> Path:
        """根据配置初始化车辆截图目录"""
        default_dir = GetVehicleScreenshotsDir()
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir
    
    def _get_ai_config(self) -> NavigationConfig:
        """获取AI配置"""
        config_path = GetConfigPath()
        if not config_path.exists():
            # 如果配置文件不存在，使用模板路径
            template_path = GetConfigTemplatePath()
            if template_path.exists():
                logger.warning(f"配置文件不存在，使用模板: {template_path}")
                config_path = template_path
            else:
                # 如果模板也不存在，抛出异常
                error_msg = f"配置文件不存在: {config_path}，请创建配置文件"
                logger.error(error_msg)
                messagebox.showerror("配置错误", error_msg)
                raise FileNotFoundError(error_msg)
        
        try:
            config = load_config(config_path, base_dir=GetProgramDir())
            return config
        except Exception as e:
            error_msg = f"加载配置文件失败: {e}"
            logger.error(error_msg)
            messagebox.showerror("配置错误", error_msg)
            raise

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
        log_dir = Path("logs")
        if log_dir.exists():
            try:
                if sys.platform == "win32":
                    os.startfile(str(log_dir))
            except Exception as e:
                messagebox.showerror("错误", f"打开日志目录失败: {e}")
        else:
            messagebox.showinfo("提示", "日志目录不存在")

    def _open_config_dir(self):
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(config_dir))
        except Exception as e:
            messagebox.showerror("错误", f"打开配置目录失败: {e}")

    def _init_debug_components(self):
        """初始化调试组件"""
        try:
            self.debug_state_machine_ = StateMachine(global_context=self.global_context_)
            self.debug_map_detector_ = MapNameDetector()
            vehicle_priority = self._get_vehicle_priority()
            self.debug_tank_selector_ = TankSelector(
                self.vehicle_screenshot_dir,
                vehicle_priority
            )
            self.debug_ui_actions_ = UIActions()
            self.debug_ai_controller_ = AIController()

            self.debug_battle_task = BattleTask(
                self.debug_tank_selector_,
                self.debug_state_machine_,
                self.debug_map_detector_,
                self.debug_ai_config_,
                self.debug_ui_actions_
            )

            self.debug_ai_config_ = self._get_ai_config()
            logger.info("调试组件初始化完成")
        except Exception as e:
            logger.error(f"调试组件初始化失败: {e}")
            messagebox.showerror("错误", f"调试组件初始化失败: {e}")

    def _build_debug_ui(self, parent):
        """构建单步调试UI"""
        # 导航AI控制区
        ai_ctrl_frame = tk.Frame(parent, bg="#F0F0F0")
        ai_ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(
            ai_ctrl_frame, text="导航AI控制：", font=("Microsoft YaHei", 9),
            bg="#F0F0F0"
        ).pack(side=tk.LEFT, padx=5)
        
        self.debug_ai_status_label = tk.Label(
            ai_ctrl_frame, text="● 已停止", font=("Microsoft YaHei", 9),
            bg="#F0F0F0", fg="#666666"
        )
        self.debug_ai_status_label.pack(side=tk.LEFT, padx=5)
        
        self.debug_ai_start_btn = tk.Button(
            ai_ctrl_frame, text="启动导航AI", command=self._debug_start_ai,
            font=("Microsoft YaHei", 9), width=12, bg="#90EE90"
        )
        self.debug_ai_start_btn.pack(side=tk.LEFT, padx=5)
        
        self.debug_ai_stop_btn = tk.Button(
            ai_ctrl_frame, text="停止导航AI", command=self._debug_stop_ai,
            font=("Microsoft YaHei", 9), width=12, bg="#FFB6C1", state=tk.DISABLED
        )
        self.debug_ai_stop_btn.pack(side=tk.LEFT, padx=5)

        # 单步测试按钮区
        test_btn_frame = tk.Frame(parent, bg="#F0F0F0")
        test_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(
            test_btn_frame, text="单步测试：", font=("Microsoft YaHei", 9, "bold"),
            bg="#F0F0F0"
        ).pack(anchor="w", padx=5, pady=2)
        
        btn_row1 = tk.Frame(test_btn_frame, bg="#F0F0F0")
        btn_row1.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(
            btn_row1, text="1. 选择坦克", command=self._debug_select_tank,
            font=("Microsoft YaHei", 9), width=15, bg="#E6E6FA"
        ).pack(side=tk.LEFT, padx=3)
        
        tk.Button(
            btn_row1, text="2. 加入战斗", command=self._debug_join_battle,
            font=("Microsoft YaHei", 9), width=15, bg="#E6E6FA"
        ).pack(side=tk.LEFT, padx=3)
        
        tk.Button(
            btn_row1, text="3. 识别地图名称", command=self._debug_detect_map,
            font=("Microsoft YaHei", 9), width=15, bg="#E6E6FA"
        ).pack(side=tk.LEFT, padx=3)
        
        btn_row2 = tk.Frame(test_btn_frame, bg="#F0F0F0")
        btn_row2.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(
            btn_row2, text="4. 返回车库", command=self._debug_return_garage,
            font=("Microsoft YaHei", 9), width=15, bg="#E6E6FA"
        ).pack(side=tk.LEFT, padx=3)
        
        tk.Button(
            btn_row2, text="检测当前状态", command=self._debug_detect_state,
            font=("Microsoft YaHei", 9), width=15, bg="#FFFACD"
        ).pack(side=tk.LEFT, padx=3)
        
        tk.Button(
            btn_row2, text="刷新调试组件", command=self._debug_refresh_components,
            font=("Microsoft YaHei", 9), width=15, bg="#F0F0F0"
        ).pack(side=tk.LEFT, padx=3)

        # 状态显示区
        status_frame = tk.Frame(parent, bg="#F0F0F0", relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Label(
            status_frame, text="调试状态信息：", font=("Microsoft YaHei", 9, "bold"),
            bg="#F0F0F0", anchor="w"
        ).pack(fill=tk.X, padx=5, pady=2)
        
        # 使用Text组件显示状态信息，支持滚动
        text_frame = tk.Frame(status_frame, bg="white")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.debug_status_text = tk.Text(
            text_frame, height=6, font=("Consolas", 9),
            wrap=tk.WORD, bg="white", fg="black"
        )
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.debug_status_text.yview)
        self.debug_status_text.configure(yscrollcommand=scrollbar.set)
        
        self.debug_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self._debug_log_status("调试组件已就绪")

    def _debug_log_status(self, message: str):
        """在调试状态区域记录消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.debug_status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.debug_status_text.see(tk.END)
        logger.info(f"[调试] {message}")

    def _debug_start_ai(self):
        """启动导航AI"""
        if self.debug_ai_running_:
            messagebox.showwarning("警告", "导航AI已在运行")
            return
        
        try:
            map_name = self.debug_map_detector_.detect()
            if not map_name:
                map_name = "default"
                self._debug_log_status(f"未识别到地图，使用默认配置: {map_name}")
            else:
                self._debug_log_status(f"识别到地图: {map_name}")
            
            if self.debug_ai_controller_.start(self.debug_ai_config_, map_name):
                self.debug_ai_running_ = True
                self.debug_ai_status_label.config(text="○ 运行中", fg="#00AA00")
                self.debug_ai_start_btn.config(state=tk.DISABLED)
                self.debug_ai_stop_btn.config(state=tk.NORMAL)
                self._debug_log_status(f"导航AI已启动 (地图: {map_name})")
            else:
                messagebox.showerror("错误", "导航AI启动失败")
                self._debug_log_status("导航AI启动失败")
        except Exception as e:
            logger.error(f"启动导航AI失败: {e}")
            messagebox.showerror("错误", f"启动导航AI失败: {e}")
            self._debug_log_status(f"启动导航AI失败: {e}")

    def _debug_stop_ai(self):
        """停止导航AI"""
        if not self.debug_ai_running_:
            return
        
        try:
            self.debug_ai_controller_.stop()
            self.debug_ai_running_ = False
            self.debug_ai_status_label.config(text="● 已停止", fg="#666666")
            self.debug_ai_start_btn.config(state=tk.NORMAL)
            self.debug_ai_stop_btn.config(state=tk.DISABLED)
            self._debug_log_status("导航AI已停止")
        except Exception as e:
            logger.error(f"停止导航AI失败: {e}")
            messagebox.showerror("错误", f"停止导航AI失败: {e}")
            self._debug_log_status(f"停止导航AI失败: {e}")

    def _debug_select_tank(self):
        """单步测试：选择坦克"""
        self._debug_log_status("开始测试：选择坦克...")
        
        if not self.debug_tank_selector_:
            messagebox.showerror("错误", "坦克选择器未初始化")
            return
        
        try:
            success = self.debug_battle_task.select_tank()
            if success:
                self._debug_log_status("✓ 成功选择坦克")
            else:
                messagebox.showwarning("警告", "未找到坦克")
                self._debug_log_status("✗ 未找到坦克")
        except Exception as e:
            logger.error(f"选择坦克失败: {e}")
            messagebox.showerror("错误", f"选择坦克失败: {e}")
            self._debug_log_status(f"选择坦克失败: {e}")

    def _debug_join_battle(self):
        """单步测试：加入战斗"""
        self._debug_log_status("开始测试：加入战斗...")
        
        try:
            success = self.debug_battle_task.enter_battle()
            if success:
                self._debug_log_status("✓ 成功加入战斗")
            else:
                messagebox.showwarning("警告", "未找到加入战斗按钮")
                self._debug_log_status("✗ 未找到加入战斗按钮")
        except Exception as e:
            logger.error(f"加入战斗失败: {e}")
            messagebox.showerror("错误", f"加入战斗失败: {e}")
            self._debug_log_status(f"加入战斗失败: {e}")

    def _debug_detect_map(self):
        """单步测试：识别地图名称"""
        self._debug_log_status("开始测试：识别地图名称...")
        
        try:
            sleep(5)
            map_name = self.debug_map_detector_.detect()
            if map_name:
                self._debug_log_status(f"✓ 从暂停界面识别到地图: {map_name}")
                messagebox.showinfo("成功", f"识别到地图: {map_name}")
                return
            messagebox.showwarning("警告", "未能识别到地图名称")
            self._debug_log_status("✗ 未能识别到地图名称")
        except Exception as e:
            logger.error(f"识别地图失败: {e}")
            messagebox.showerror("错误", f"识别地图失败: {e}")
            self._debug_log_status(f"识别地图失败: {e}")

    def _debug_return_garage(self):
        """单步测试：返回车库"""
        self._debug_log_status("开始测试：返回车库...")
        try:
            sleep(5)
            success = self.debug_battle_task.enter_garage()
            if success:
                self._debug_log_status("✓ 成功返回车库")
            else:
                messagebox.showwarning("警告", "未找到返回车库按钮")
                self._debug_log_status("✗ 未找到返回车库按钮")
        except Exception as e:
            logger.error(f"返回车库失败: {e}")
            messagebox.showerror("错误", f"返回车库失败: {e}")
            self._debug_log_status(f"返回车库失败: {e}")

    def _debug_detect_state(self):
        """检测当前游戏状态"""
        self._debug_log_status("开始检测当前游戏状态...")

        if not self.debug_state_machine_:
            messagebox.showerror("错误", "状态机未初始化")
            return

        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="选择要检测的图片",
            filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")]
        )
        if not file_path:
            self._debug_log_status("✗ 未选择任何图片")
            messagebox.showwarning("警告", "未选择任何图片")
            return

        try:
            frame = cv2.imread(file_path)
            for _ in range(3):
                self.debug_state_machine_.update(frame=frame)
            self._debug_log_status(f"✓ 已加载图片: {file_path}")
            current_state = self.debug_state_machine_.current_state()
            state_text = f"当前游戏状态: {current_state.value}"
            self._debug_log_status(state_text)
            messagebox.showinfo("状态检测", state_text)
        except Exception as e:
            msg = f"检测状态失败: {e}"
            logger.error(msg)
            messagebox.showerror("错误", msg)
            self._debug_log_status(msg)

    def _debug_refresh_components(self):
        """刷新调试组件"""
        self._debug_log_status("刷新调试组件...")
        try:
            self._init_debug_components()
            self._debug_log_status("✓ 调试组件刷新完成")
            messagebox.showinfo("成功", "调试组件刷新完成")
        except Exception as e:
            logger.error(f"刷新调试组件失败: {e}")
            messagebox.showerror("错误", f"刷新调试组件失败: {e}")
            self._debug_log_status(f"刷新调试组件失败: {e}")

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