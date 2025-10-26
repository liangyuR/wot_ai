"""
简单的配置 GUI - 让非技术用户轻松设置参数
使用 tkinter (Python 内置，无需额外依赖)
"""
import tkinter as tk
from tkinter import ttk, messagebox
import yaml
import sys
import os
from pathlib import Path


def get_base_path():
    """获取程序基础路径（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的 exe
        return Path(sys.executable).parent
    else:
        # 如果是 Python 脚本
        return Path(__file__).parent


class ConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("坦克世界 AI - 数据采集配置")
        self.root.geometry("550x550")
        self.root.resizable(True, True)
        self.root.minsize(550, 600)  # 最小尺寸
        
        # 配置文件路径（兼容打包后的 exe）
        base_path = get_base_path()
        self.config_path = base_path / "configs" / "client_config.yaml"
        
        # 确保配置目录存在
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载当前配置
        self.load_config()
        
        # 创建界面
        self.create_widgets()
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("错误", f"无法加载配置文件: {e}")
            self.config = {
                'capture': {
                    'fps': 5,
                    'mode': 'fullscreen',
                    'fullscreen': {'width': 1920, 'height': 1080}
                }
            }
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            return True
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")
            return False
    
    def create_widgets(self):
        """创建界面组件"""
        # 标题
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=70)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🎮 坦克世界 AI 数据采集工具",
            font=("微软雅黑", 16, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # 主配置区域
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. 分辨率预设
        ttk.Label(main_frame, text="屏幕分辨率:", font=("微软雅黑", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5)
        )
        
        resolution_frame = ttk.Frame(main_frame)
        resolution_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        self.resolution_var = tk.StringVar()
        current_w = self.config.get('capture', {}).get('fullscreen', {}).get('width', 1920)
        current_h = self.config.get('capture', {}).get('fullscreen', {}).get('height', 1080)
        current_res = f"{current_w}x{current_h}"
        
        resolutions = [
            "1920x1080 (Full HD)",
            "2560x1440 (2K)",
            "3440x1440 (超宽屏)",
            "3840x2160 (4K)"
        ]
        
        # 尝试匹配当前分辨率
        matched = False
        for res in resolutions:
            if current_res in res:
                self.resolution_var.set(res)
                matched = True
                break
        if not matched:
            self.resolution_var.set(f"{current_res} (自定义)")
        
        for i, res in enumerate(resolutions):
            ttk.Radiobutton(
                resolution_frame,
                text=res,
                variable=self.resolution_var,
                value=res
            ).grid(row=i // 2, column=i % 2, sticky=tk.W, padx=(0, 20), pady=2)
        
        # 2. FPS 设置
        ttk.Label(main_frame, text="录制帧率 (FPS):", font=("微软雅黑", 10, "bold")).grid(
            row=2, column=0, sticky=tk.W, pady=(15, 5)
        )
        
        fps_frame = ttk.Frame(main_frame)
        fps_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        self.fps_var = tk.IntVar(value=self.config.get('capture', {}).get('fps', 5))
        
        fps_options = [
            (5, "5 FPS (推荐 - 省空间)"),
            (10, "10 FPS (平衡)"),
            (15, "15 FPS (更流畅)")
        ]
        
        for fps, label in fps_options:
            ttk.Radiobutton(
                fps_frame,
                text=label,
                variable=self.fps_var,
                value=fps
            ).pack(anchor=tk.W, pady=2)
        
        # 3. 存储估算
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=15
        )
        
        self.info_label = tk.Label(
            main_frame,
            text="",
            font=("Consolas", 9),
            justify=tk.LEFT,
            bg="#ecf0f1",
            fg="#34495e",
            padx=10,
            pady=10
        )
        self.info_label.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        
        self.update_info()
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="💾 保存配置",
            command=self.apply_config,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🎬 开始录制",
            command=self.start_recording,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="❌ 退出",
            command=self.root.quit,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # 绑定变量改变事件
        self.resolution_var.trace('w', lambda *args: self.update_info())
        self.fps_var.trace('w', lambda *args: self.update_info())
    
    def update_info(self):
        """更新存储估算信息"""
        # 解析分辨率
        res_str = self.resolution_var.get().split()[0]
        try:
            width, height = map(int, res_str.split('x'))
        except:
            width, height = 1920, 1080
            
        fps = self.fps_var.get()
        
        # 估算存储（JPEG 压缩后约 0.1-0.3 bits per pixel）
        pixels = width * height
        bytes_per_frame = pixels * 0.2  # 平均压缩率
        
        # 每分钟 (考虑 frame_step=2)
        actual_fps = fps / 2
        mb_per_minute = (bytes_per_frame * actual_fps * 60) / (1024 * 1024)
        
        # 每小时
        mb_per_hour = mb_per_minute * 60
        
        info_text = f"""📊 存储估算 (frame_step=2):

分辨率: {width}x{height}
录制帧率: {fps} FPS
实际保存: {actual_fps:.1f} FPS

每分钟: ~{mb_per_minute:.1f} MB
每小时: ~{mb_per_hour / 1024:.2f} GB
10 场战斗 (约 2 小时): ~{mb_per_hour * 2 / 1024:.2f} GB"""
        
        self.info_label.config(text=info_text)
    
    def apply_config(self):
        """应用配置"""
        # 解析分辨率
        res_str = self.resolution_var.get().split()[0]
        try:
            width, height = map(int, res_str.split('x'))
        except:
            messagebox.showerror("错误", "无法解析分辨率，请重新选择")
            return
            
        fps = self.fps_var.get()
        
        # 更新配置
        if 'capture' not in self.config:
            self.config['capture'] = {}
        if 'fullscreen' not in self.config['capture']:
            self.config['capture']['fullscreen'] = {}
        
        self.config['capture']['fullscreen']['width'] = width
        self.config['capture']['fullscreen']['height'] = height
        self.config['capture']['fps'] = fps
        
        # 保存
        if self.save_config():
            messagebox.showinfo("成功", "配置已保存！\n\n可以开始录制了。")
    
    def start_recording(self):
        """启动录制"""
        # 先保存配置
        res_str = self.resolution_var.get().split()[0]
        try:
            width, height = map(int, res_str.split('x'))
        except:
            messagebox.showerror("错误", "无法解析分辨率，请重新选择")
            return
            
        fps = self.fps_var.get()
        
        if 'capture' not in self.config:
            self.config['capture'] = {}
        if 'fullscreen' not in self.config['capture']:
            self.config['capture']['fullscreen'] = {}
        
        self.config['capture']['fullscreen']['width'] = width
        self.config['capture']['fullscreen']['height'] = height
        self.config['capture']['fps'] = fps
        
        if not self.save_config():
            return
            
        messagebox.showinfo(
            "启动录制",
            "配置已保存！\n\n将启动录制程序...\n\n快捷键说明：\n"
            "• F9  - 开始录制\n"
            "• F10 - 停止录制\n"
            "• Ctrl+C - 退出程序\n\n"
            "使用方法：\n"
            "1. 进入游戏战斗\n"
            "2. 按 F9 开始录制\n"
            "3. 正常游戏\n"
            "4. 按 F10 停止录制\n"
            "5. 可重复按 F9/F10 录制多场\n"
            "6. 按 Ctrl+C 或关闭窗口退出"
        )
        
        # 直接在当前进程中启动录制（支持单 exe 打包）
        self.root.withdraw()  # 隐藏 GUI 窗口
        
        try:
            # 导入录制模块
            sys.path.insert(0, str(get_base_path()))
            from data_collection.record_gameplay import main as record_main
            
            # 启动录制
            record_main()
            
        except Exception as e:
            messagebox.showerror("错误", f"录制过程出错: {e}\n\n详情请查看日志")
            import traceback
            traceback.print_exc()
        finally:
            # 录制结束后关闭程序
            self.root.quit()


def main():
    root = tk.Tk()
    app = ConfigGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

