#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI主界面：瞄准辅助系统配置和管理
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import logging
import threading

# 统一导入方式
try:
    from yolo.controller import AimAssistMainController
    from yolo.core.config_manager import AimConfigManager
except ImportError:
    # 回退：如果 yolo 包不可用，使用相对导入
    from controller import AimAssistMainController
    from core.config_manager import AimConfigManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "aim_config.yaml"


class AimAssistGUI:
    """瞄准辅助系统GUI"""
    
    def __init__(self, root):
        """初始化GUI"""
        self.root = root
        self.root.title("YOLO 瞄准辅助系统")
        self.root.geometry("700x800")
        self.root.resizable(True, True)
        self.root.minsize(650, 750)
        
        # 配置管理器
        self.config_manager_ = AimConfigManager(CONFIG_PATH)
        self.config_ = self.config_manager_.Load()
        
        # 主控制器
        self.controller_ = None
        
        # 热键捕获状态
        self.capturing_hotkey_ = None
        
        # 创建界面
        self.CreateWidgets()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.OnExit)
        
        # 更新界面数据
        self.LoadConfigToUI()
    
    def CreateWidgets(self):
        """创建界面组件"""
        # 标题
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🎯 YOLO 瞄准辅助系统",
            font=("微软雅黑", 16, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)
        
        # 主容器（使用Notebook实现标签页）
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 配置标签页
        config_frame = ttk.Frame(notebook, padding=15)
        notebook.add(config_frame, text="配置")
        self.CreateConfigTab(config_frame)
        
        # 状态标签页
        status_frame = ttk.Frame(notebook, padding=15)
        notebook.add(status_frame, text="状态")
        self.CreateStatusTab(status_frame)
        
        # 路径规划标签页
        path_planning_frame = ttk.Frame(notebook, padding=15)
        notebook.add(path_planning_frame, text="路径规划")
        self.CreatePathPlanningTab(path_planning_frame)
        
        # 底部按钮
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(
            button_frame,
            text="▶ 启动",
            command=self.StartController,
            width=12
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="⏹ 停止",
            command=self.StopController,
            width=12,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="💾 保存配置",
            command=self.SaveConfig,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🔄 加载配置",
            command=self.LoadConfig,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="❌ 退出",
            command=self.OnExit,
            width=12
        ).pack(side=tk.RIGHT, padx=5)
    
    def CreateConfigTab(self, parent):
        """创建配置标签页"""
        # 创建滚动区域
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # 屏幕设置
        screen_frame = ttk.LabelFrame(scrollable_frame, text="屏幕设置", padding=10)
        screen_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(screen_frame, text="宽度:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.screen_width_var = tk.StringVar(value="1920")
        ttk.Entry(screen_frame, textvariable=self.screen_width_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(screen_frame, text="高度:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.screen_height_var = tk.StringVar(value="1080")
        ttk.Entry(screen_frame, textvariable=self.screen_height_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(
            screen_frame,
            text="自动检测",
            command=self.AutoDetectResolution
        ).grid(row=0, column=4, padx=5, pady=5)
        
        # FOV 设置
        fov_frame = ttk.LabelFrame(scrollable_frame, text="FOV 设置", padding=10)
        fov_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(fov_frame, text="水平 FOV (度):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.h_fov_var = tk.StringVar(value="90.0")
        ttk.Entry(fov_frame, textvariable=self.h_fov_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        self.auto_v_fov_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            fov_frame,
            text="自动计算垂直 FOV",
            variable=self.auto_v_fov_var,
            command=self.ToggleAutoVFOV
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(fov_frame, text="垂直 FOV (度):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.v_fov_var = tk.StringVar(value="")
        self.v_fov_entry = ttk.Entry(fov_frame, textvariable=self.v_fov_var, width=10, state=tk.DISABLED)
        self.v_fov_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # 鼠标设置
        mouse_frame = ttk.LabelFrame(scrollable_frame, text="鼠标设置", padding=10)
        mouse_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(mouse_frame, text="灵敏度:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.mouse_sensitivity_var = tk.StringVar(value="1.0")
        ttk.Entry(mouse_frame, textvariable=self.mouse_sensitivity_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(mouse_frame, text="标定系数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.calibration_factor_var = tk.StringVar(value="")
        ttk.Entry(mouse_frame, textvariable=self.calibration_factor_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(
            mouse_frame,
            text="标定工具",
            command=self.OpenCalibrationTool
        ).grid(row=1, column=2, padx=5, pady=5)
        
        # 平滑参数
        smoothing_frame = ttk.LabelFrame(scrollable_frame, text="平滑参数", padding=10)
        smoothing_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(smoothing_frame, text="平滑系数 (0.0-1.0):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.smoothing_factor_var = tk.DoubleVar(value=0.3)
        smoothing_scale = ttk.Scale(
            smoothing_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.smoothing_factor_var,
            length=200
        )
        smoothing_scale.grid(row=0, column=1, padx=5, pady=5)
        
        self.smoothing_factor_label = ttk.Label(smoothing_frame, text="0.3")
        self.smoothing_factor_label.grid(row=0, column=2, padx=5, pady=5)
        smoothing_scale.configure(command=lambda v: self.smoothing_factor_label.config(text=f"{float(v):.2f}"))
        
        ttk.Label(smoothing_frame, text="最大步长:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_step_var = tk.StringVar(value="50.0")
        ttk.Entry(smoothing_frame, textvariable=self.max_step_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # 热键设置
        hotkey_frame = ttk.LabelFrame(scrollable_frame, text="热键设置", padding=10)
        hotkey_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(hotkey_frame, text="激活/禁用热键:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.toggle_hotkey_var = tk.StringVar(value="f8")
        self.toggle_hotkey_entry = ttk.Entry(hotkey_frame, textvariable=self.toggle_hotkey_var, width=10, state="readonly")
        self.toggle_hotkey_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.toggle_capture_button = ttk.Button(
            hotkey_frame,
            text="捕获",
            command=lambda: self.StartCaptureHotkey('toggle')
        )
        self.toggle_capture_button.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(hotkey_frame, text="退出热键:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.exit_hotkey_var = tk.StringVar(value="esc")
        self.exit_hotkey_entry = ttk.Entry(hotkey_frame, textvariable=self.exit_hotkey_var, width=10, state="readonly")
        self.exit_hotkey_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.exit_capture_button = ttk.Button(
            hotkey_frame,
            text="捕获",
            command=lambda: self.StartCaptureHotkey('exit')
        )
        self.exit_capture_button.grid(row=1, column=2, padx=5, pady=5)
        
        # 检测设置
        detection_frame = ttk.LabelFrame(scrollable_frame, text="检测设置", padding=10)
        detection_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(detection_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_path_var = tk.StringVar(value="train/model/yolo11n.pt")
        ttk.Entry(detection_frame, textvariable=self.model_path_var, width=30).grid(row=0, column=1, padx=5, pady=5, columnspan=2)
        
        ttk.Button(
            detection_frame,
            text="选择",
            command=self.SelectModelFile
        ).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(detection_frame, text="目标类别 ID:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_class_var = tk.StringVar(value="")
        ttk.Entry(detection_frame, textvariable=self.target_class_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(detection_frame, text="(留空则选择置信度最高的)", font=("微软雅黑", 8)).grid(row=1, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(detection_frame, text="检测 FPS:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.fps_var = tk.StringVar(value="30.0")
        ttk.Entry(detection_frame, textvariable=self.fps_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # 配置列权重
        scrollable_frame.columnconfigure(0, weight=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def CreateStatusTab(self, parent):
        """创建状态标签页"""
        # 状态显示
        status_frame = ttk.LabelFrame(parent, text="运行状态", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.status_label = tk.Label(
            status_frame,
            text="状态: 未启动",
            font=("微软雅黑", 12, "bold"),
            fg="#e74c3c"
        )
        self.status_label.pack(pady=10)
        
        # 统计信息
        stats_frame = ttk.LabelFrame(parent, text="统计信息", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.stats_text = tk.Text(
            stats_frame,
            height=15,
            font=("Consolas", 10),
            wrap=tk.WORD
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text.insert("1.0", "等待启动...\n")
        self.stats_text.config(state=tk.DISABLED)
    
    def CreatePathPlanningTab(self, parent):
        """创建路径规划标签页"""
        # 路径规划控制器
        self.path_planning_controller_ = None
        
        # 配置区域
        config_frame = ttk.LabelFrame(parent, text="路径规划配置", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 模型路径
        ttk.Label(config_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.path_model_path_var = tk.StringVar(value="path_planning/config/path_planning_config.yaml")
        ttk.Entry(config_frame, textvariable=self.path_model_path_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(
            config_frame,
            text="选择配置",
            command=self.SelectPathPlanningConfig
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # 小地图区域设置
        minimap_frame = ttk.LabelFrame(config_frame, text="小地图区域", padding=5)
        minimap_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(minimap_frame, text="X:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.minimap_x_var = tk.StringVar(value="1600")
        ttk.Entry(minimap_frame, textvariable=self.minimap_x_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(minimap_frame, text="Y:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.minimap_y_var = tk.StringVar(value="800")
        ttk.Entry(minimap_frame, textvariable=self.minimap_y_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(minimap_frame, text="宽度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.minimap_width_var = tk.StringVar(value="320")
        ttk.Entry(minimap_frame, textvariable=self.minimap_width_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(minimap_frame, text="高度:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.minimap_height_var = tk.StringVar(value="320")
        ttk.Entry(minimap_frame, textvariable=self.minimap_height_var, width=10).grid(row=1, column=3, padx=5, pady=2)
        
        # 控制按钮
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.path_start_button = ttk.Button(
            button_frame,
            text="▶ 启动路径规划",
            command=self.StartPathPlanning,
            width=15
        )
        self.path_start_button.pack(side=tk.LEFT, padx=5)
        
        self.path_stop_button = ttk.Button(
            button_frame,
            text="⏹ 停止",
            command=self.StopPathPlanning,
            width=15,
            state=tk.DISABLED
        )
        self.path_stop_button.pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        status_frame = ttk.LabelFrame(parent, text="路径规划状态", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.path_status_label = tk.Label(
            status_frame,
            text="状态: 未启动",
            font=("微软雅黑", 12, "bold"),
            fg="#e74c3c"
        )
        self.path_status_label.pack(pady=10)
        
        # 统计信息
        self.path_stats_text = tk.Text(
            status_frame,
            height=10,
            font=("Consolas", 10),
            wrap=tk.WORD
        )
        self.path_stats_text.pack(fill=tk.BOTH, expand=True)
        self.path_stats_text.insert("1.0", "等待启动...\n")
        self.path_stats_text.config(state=tk.DISABLED)
    
    def SelectPathPlanningConfig(self):
        """选择路径规划配置文件"""
        filename = filedialog.askopenfilename(
            title="选择路径规划配置文件",
            filetypes=[("YAML文件", "*.yaml"), ("所有文件", "*.*")],
            initialdir=str(Path(__file__).resolve().parent / "path_planning" / "config")
        )
        if filename:
            try:
                rel_path = Path(filename).relative_to(Path(__file__).resolve().parent)
                self.path_model_path_var.set(str(rel_path))
            except ValueError:
                self.path_model_path_var.set(filename)
    
    def StartPathPlanning(self):
        """启动路径规划"""
        if self.path_planning_controller_ and self.path_planning_controller_.IsRunning():
            messagebox.showwarning("警告", "路径规划已在运行")
            return
        
        try:
            try:
                from path_planning.controller.path_planning_controller import PathPlanningController
            except ImportError:
                # 备用导入路径
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from path_planning.controller.path_planning_controller import PathPlanningController
            
            config_path = Path(self.path_model_path_var.get())
            if not config_path.is_absolute():
                config_path = Path(__file__).resolve().parent / config_path
            
            if not config_path.exists():
                messagebox.showerror("错误", f"配置文件不存在: {config_path}")
                return
            
            # 创建控制器
            self.path_planning_controller_ = PathPlanningController(config_path)
            
            # 设置回调
            self.path_planning_controller_.SetStatusCallback(self.OnPathPlanningStatusUpdate)
            self.path_planning_controller_.SetStatsCallback(self.OnPathPlanningStatsUpdate)
            
            # 启动
            if self.path_planning_controller_.Start():
                self.path_start_button.config(state=tk.DISABLED)
                self.path_stop_button.config(state=tk.NORMAL)
                messagebox.showinfo("成功", "路径规划已启动")
            else:
                messagebox.showerror("错误", "路径规划启动失败")
        except Exception as e:
            messagebox.showerror("错误", f"启动失败: {e}")
            logger.error(f"启动路径规划失败: {e}")
    
    def StopPathPlanning(self):
        """停止路径规划"""
        if not self.path_planning_controller_ or not self.path_planning_controller_.IsRunning():
            return
        
        try:
            self.path_planning_controller_.Stop()
            self.path_start_button.config(state=tk.NORMAL)
            self.path_stop_button.config(state=tk.DISABLED)
            messagebox.showinfo("成功", "路径规划已停止")
        except Exception as e:
            messagebox.showerror("错误", f"停止失败: {e}")
    
    def OnPathPlanningStatusUpdate(self, status: str):
        """路径规划状态更新回调"""
        def update():
            if status == "运行中":
                self.path_status_label.config(text=f"状态: {status}", fg="#27ae60")
            elif status == "已停止":
                self.path_status_label.config(text=f"状态: {status}", fg="#e74c3c")
            else:
                self.path_status_label.config(text=f"状态: {status}", fg="#3498db")
        
        self.root.after(0, update)
    
    def OnPathPlanningStatsUpdate(self, stats: dict):
        """路径规划统计信息更新回调"""
        def update():
            self.path_stats_text.config(state=tk.NORMAL)
            self.path_stats_text.delete("1.0", tk.END)
            
            stats_str = f"""FPS: {stats.get('fps', 0):.1f}
总帧数: {stats.get('frame_count', 0)}
检测次数: {stats.get('detection_count', 0)}
路径规划次数: {stats.get('path_planning_count', 0)}
导航执行次数: {stats.get('navigation_count', 0)}
"""
            self.path_stats_text.insert("1.0", stats_str)
            self.path_stats_text.config(state=tk.DISABLED)
        
        self.root.after(0, update)
    
    def AutoDetectResolution(self):
        """自动检测屏幕分辨率"""
        try:
            import mss
            sct = mss.mss()
            monitor_info = sct.monitors[1]
            width = monitor_info['width']
            height = monitor_info['height']
            
            self.screen_width_var.set(str(width))
            self.screen_height_var.set(str(height))
            
            messagebox.showinfo("成功", f"已检测到分辨率: {width}x{height}")
        except Exception as e:
            messagebox.showerror("错误", f"无法检测分辨率: {e}")
    
    def ToggleAutoVFOV(self):
        """切换自动计算垂直FOV"""
        if self.auto_v_fov_var.get():
            self.v_fov_entry.config(state=tk.DISABLED)
            self.v_fov_var.set("")
        else:
            self.v_fov_entry.config(state=tk.NORMAL)
    
    def StartCaptureHotkey(self, hotkey_name: str):
        """开始捕获热键"""
        self.capturing_hotkey_ = hotkey_name
        
        if hotkey_name == 'toggle':
            self.toggle_capture_button.config(text="按下按键...", state=tk.DISABLED)
            self.toggle_hotkey_entry.config(state=tk.NORMAL)
        else:
            self.exit_capture_button.config(text="按下按键...", state=tk.DISABLED)
            self.exit_hotkey_entry.config(state=tk.NORMAL)
        
        # 绑定全局按键事件
        self.root.bind_all("<KeyPress>", self.OnHotkeyCapture)
        messagebox.showinfo("提示", "请按下要设置的按键")
    
    def OnHotkeyCapture(self, event):
        """捕获热键事件"""
        if self.capturing_hotkey_ is None:
            return
        
        # 获取按键名称
        key_name = event.keysym.lower()
        
        # 特殊键处理
        special_keys = {
            'escape': 'esc',
            'return': 'enter',
            'space': 'space',
        }
        key_name = special_keys.get(key_name, key_name)
        
        if self.capturing_hotkey_ == 'toggle':
            self.toggle_hotkey_var.set(key_name)
            self.toggle_capture_button.config(text="捕获", state=tk.NORMAL)
            self.toggle_hotkey_entry.config(state="readonly")
        else:
            self.exit_hotkey_var.set(key_name)
            self.exit_capture_button.config(text="捕获", state=tk.NORMAL)
            self.exit_hotkey_entry.config(state="readonly")
        
        self.root.unbind_all("<KeyPress>")
        self.capturing_hotkey_ = None
    
    def SelectModelFile(self):
        """选择模型文件"""
        filename = filedialog.askopenfilename(
            title="选择YOLO模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if filename:
            # 转换为相对路径（如果可能）
            try:
                rel_path = Path(filename).relative_to(SCRIPT_DIR)
                self.model_path_var.set(str(rel_path))
            except ValueError:
                self.model_path_var.set(filename)
    
    def OpenCalibrationTool(self):
        """打开标定工具"""
        messagebox.showinfo("提示", "请运行 calibration.py 进行标定")
    
    def LoadConfigToUI(self):
        """从配置加载到UI"""
        self.screen_width_var.set(str(self.config_.get('screen', {}).get('width', 1920)))
        self.screen_height_var.set(str(self.config_.get('screen', {}).get('height', 1080)))
        
        self.h_fov_var.set(str(self.config_.get('fov', {}).get('horizontal', 90.0)))
        v_fov = self.config_.get('fov', {}).get('vertical')
        if v_fov is None:
            self.auto_v_fov_var.set(True)
            self.v_fov_var.set("")
            self.v_fov_entry.config(state=tk.DISABLED)
        else:
            self.auto_v_fov_var.set(False)
            self.v_fov_var.set(str(v_fov))
            self.v_fov_entry.config(state=tk.NORMAL)
        
        self.mouse_sensitivity_var.set(str(self.config_.get('mouse', {}).get('sensitivity', 1.0)))
        calibration_factor = self.config_.get('mouse', {}).get('calibration_factor')
        self.calibration_factor_var.set(str(calibration_factor) if calibration_factor else "")
        
        self.smoothing_factor_var.set(self.config_.get('smoothing', {}).get('factor', 0.3))
        self.max_step_var.set(str(self.config_.get('smoothing', {}).get('max_step', 50.0)))
        
        self.toggle_hotkey_var.set(self.config_.get('hotkeys', {}).get('toggle', 'f8'))
        self.exit_hotkey_var.set(self.config_.get('hotkeys', {}).get('exit', 'esc'))
        
        self.model_path_var.set(self.config_.get('detection', {}).get('model_path', 'train/model/yolo11n.pt'))
        target_class = self.config_.get('detection', {}).get('target_class')
        self.target_class_var.set(str(target_class) if target_class is not None else "")
        self.fps_var.set(str(self.config_.get('detection', {}).get('fps', 30.0)))
    
    def SaveConfigFromUI(self) -> dict:
        """从UI保存到配置字典"""
        config = {}
        
        config['screen'] = {
            'width': int(self.screen_width_var.get()),
            'height': int(self.screen_height_var.get())
        }
        
        config['fov'] = {
            'horizontal': float(self.h_fov_var.get())
        }
        if not self.auto_v_fov_var.get() and self.v_fov_var.get():
            config['fov']['vertical'] = float(self.v_fov_var.get())
        else:
            config['fov']['vertical'] = None
        
        config['mouse'] = {
            'sensitivity': float(self.mouse_sensitivity_var.get())
        }
        calibration_factor = self.calibration_factor_var.get()
        if calibration_factor:
            config['mouse']['calibration_factor'] = float(calibration_factor)
        else:
            config['mouse']['calibration_factor'] = None
        
        config['smoothing'] = {
            'factor': self.smoothing_factor_var.get(),
            'max_step': float(self.max_step_var.get())
        }
        
        config['hotkeys'] = {
            'toggle': self.toggle_hotkey_var.get(),
            'exit': self.exit_hotkey_var.get()
        }
        
        config['detection'] = {
            'model_path': self.model_path_var.get(),
            'fps': float(self.fps_var.get())
        }
        target_class = self.target_class_var.get()
        if target_class:
            config['detection']['target_class'] = int(target_class)
        else:
            config['detection']['target_class'] = None
        
        return config
    
    def SaveConfig(self):
        """保存配置"""
        try:
            config = self.SaveConfigFromUI()
            self.config_manager_.Save(config)
            self.config_ = config
            messagebox.showinfo("成功", "配置已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")
    
    def LoadConfig(self):
        """加载配置"""
        self.config_ = self.config_manager_.Load()
        self.LoadConfigToUI()
        messagebox.showinfo("成功", "配置已加载")
    
    def StartController(self):
        """启动控制器"""
        if self.controller_ and self.controller_.IsRunning():
            messagebox.showwarning("警告", "控制器已在运行")
            return
        
        try:
            # 保存当前配置
            config = self.SaveConfigFromUI()
            self.config_manager_.Save(config)
            
            # 创建控制器
            self.controller_ = AimAssistMainController(CONFIG_PATH)
            
            # 设置回调
            self.controller_.SetStatusCallback(self.OnStatusUpdate)
            self.controller_.SetStatsCallback(self.OnStatsUpdate)
            
            # 启动
            self.controller_.Start()
            
            # 更新UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="状态: 运行中", fg="#27ae60")
            
            messagebox.showinfo("成功", "控制器已启动")
        except Exception as e:
            messagebox.showerror("错误", f"启动失败: {e}")
            logger.error(f"启动控制器失败: {e}")
    
    def StopController(self):
        """停止控制器"""
        if not self.controller_ or not self.controller_.IsRunning():
            return
        
        try:
            self.controller_.Stop()
            
            # 更新UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="状态: 已停止", fg="#e74c3c")
            
            messagebox.showinfo("成功", "控制器已停止")
        except Exception as e:
            messagebox.showerror("错误", f"停止失败: {e}")
    
    def OnStatusUpdate(self, status: str):
        """状态更新回调"""
        def update():
            if status == "运行中":
                self.status_label.config(text=f"状态: {status}", fg="#27ae60")
            elif status == "已停止":
                self.status_label.config(text=f"状态: {status}", fg="#e74c3c")
            else:
                self.status_label.config(text=f"状态: {status}", fg="#3498db")
        
        self.root.after(0, update)
    
    def OnStatsUpdate(self, stats: dict):
        """统计信息更新回调"""
        def update():
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete("1.0", tk.END)
            
            stats_str = f"""FPS: {stats.get('fps', 0):.1f}
总帧数: {stats.get('frame_count', 0)}
检测数量: {stats.get('detection_count', 0)}
启用状态: {'是' if self.controller_ and self.controller_.IsEnabled() else '否'}
"""
            self.stats_text.insert("1.0", stats_str)
            self.stats_text.config(state=tk.DISABLED)
        
        self.root.after(0, update)
    
    def OnExit(self):
        """退出"""
        need_confirm = False
        
        if self.controller_ and self.controller_.IsRunning():
            need_confirm = True
        
        if hasattr(self, 'path_planning_controller_') and self.path_planning_controller_ and self.path_planning_controller_.IsRunning():
            need_confirm = True
        
        if need_confirm:
            if messagebox.askyesno("确认", "有控制器正在运行，是否退出？"):
                if self.controller_ and self.controller_.IsRunning():
                    self.controller_.Stop()
                if hasattr(self, 'path_planning_controller_') and self.path_planning_controller_ and self.path_planning_controller_.IsRunning():
                    self.path_planning_controller_.Stop()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """主函数"""
    root = tk.Tk()
    app = AimAssistGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()