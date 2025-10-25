"""
Record gameplay data for imitation learning
"""

import argparse
import cv2
import numpy as np
import time
import json
from pathlib import Path
from pynput import keyboard, mouse
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))

# C++ 绑定
try:
    from cpp_bindings import ScreenCapture
    CPP_AVAILABLE = True
except ImportError:
    logger.warning("C++ bindings not available, using fallback")
    CPP_AVAILABLE = False
    import mss

# 窗口捕获
try:
    from utils.window_capture import WindowCapture
    WINDOW_CAPTURE_AVAILABLE = True
except ImportError:
    logger.warning("Window capture not available")
    WINDOW_CAPTURE_AVAILABLE = False

# 异步帧保存器
try:
    from data_collection.async_frame_saver import AsyncFrameSaver
    ASYNC_SAVER_AVAILABLE = True
except ImportError:
    try:
        from async_frame_saver import AsyncFrameSaver
        ASYNC_SAVER_AVAILABLE = True
    except ImportError:
        logger.warning("异步帧保存器不可用，将使用同步保存（可能导致卡顿）")
        ASYNC_SAVER_AVAILABLE = False


class GameplayRecorder:
    """Record gameplay footage and input data"""
    
    def __init__(
        self,
        output_dir: str = "data/sessions",
        fps: int = 30,
        capture_mode: str = "fullscreen",
        process_name: str = None,
        window_title: str = None,
        save_format: str = "frames",
        frame_step: int = 1
    ):
        """
        Initialize recorder
        
        Args:
            output_dir: Output directory for sessions
            fps: Recording FPS
            capture_mode: 'window' or 'fullscreen'
            process_name: Process name for window capture (e.g., 'WorldOfTanks.exe')
            window_title: Window title for window capture (e.g., 'World of Tanks')
            save_format: 'frames' (save JPEG frames), 'video' (encode video), 'both'
            frame_step: Save every N frames (1 = save all, 5 = save every 5th frame)
        """
        self.output_dir_ = Path(output_dir)
        self.output_dir_.mkdir(parents=True, exist_ok=True)
        
        self.fps_ = fps
        self.frame_interval_ = 1.0 / fps
        self.capture_mode_ = capture_mode
        self.save_format_ = save_format
        self.frame_step_ = frame_step
        
        # Recording state
        self.recording_ = False
        self.saved_ = False  # 是否已保存
        self.session_dir_ = None  # 当前会话目录
        self.frames_dir_ = None  # 帧保存目录
        self.frames_ = []  # 仅当 save_format 包含 'video' 时使用
        self.actions_ = []
        self.frame_numbers_ = []  # 记录每个 action 对应的帧号
        
        # Input tracking
        self.current_keys_ = set()
        self.mouse_position_ = (0, 0)
        
        # Capture objects
        self.window_capture_ = None
        self.screen_capture_ = None
        self.sct_ = None
        self.async_saver_ = None  # 异步帧保存器
        
        # Initialize screen capture
        logger.info("初始化屏幕捕获模块...")
        logger.info(f"  - 捕获模式: {capture_mode}")
        
        if capture_mode == "window":
            # 窗口捕获模式
            if not WINDOW_CAPTURE_AVAILABLE:
                logger.error("✗ 窗口捕获不可用，请安装: pip install pywin32 psutil")
                logger.info("  降级到全屏捕获模式")
                capture_mode = "fullscreen"
            else:
                try:
                    if not process_name and not window_title:
                        logger.warning("未指定进程名或窗口标题，使用默认值")
                        process_name = "WorldOfTanks.exe"
                    
                    if process_name:
                        logger.info(f"  - 通过进程名查找: {process_name}")
                        self.window_capture_ = WindowCapture(
                            process_name=process_name,
                            partial_match=True
                        )
                    else:
                        logger.info(f"  - 通过窗口标题查找: {window_title}")
                        self.window_capture_ = WindowCapture(
                            window_title=window_title,
                            partial_match=True
                        )
                    
                    if self.window_capture_.IsValid():
                        logger.info("✓ 窗口捕获模块初始化成功")
                        self.capture_mode_ = "window"
                    else:
                        logger.error("✗ 未找到目标窗口，降级到全屏模式")
                        self.window_capture_ = None
                        capture_mode = "fullscreen"
                        
                except Exception as e:
                    logger.error(f"✗ 窗口捕获初始化失败: {e}")
                    logger.info("  降级到全屏捕获模式")
                    self.window_capture_ = None
                    capture_mode = "fullscreen"
        
        if capture_mode == "fullscreen":
            # 全屏捕获模式
            if CPP_AVAILABLE:
                try:
                    self.screen_capture_ = ScreenCapture(1920, 1080)
                    logger.info("✓ C++ 屏幕捕获模块初始化成功")
                    self.capture_mode_ = "fullscreen_cpp"
                except Exception as e:
                    logger.error(f"✗ C++ 屏幕捕获初始化失败: {e}")
                    raise
            else:
                try:
                    self.sct_ = mss.mss()
                    self.monitor_ = {"top": 0, "left": 0, "width": 1920, "height": 1080}
                    logger.info("✓ Python fallback (mss) 屏幕捕获初始化成功")
                    self.capture_mode_ = "fullscreen_mss"
                except Exception as e:
                    logger.error(f"✗ mss 初始化失败: {e}")
                    logger.error("请安装 mss: pip install mss")
                    raise
        
        # Setup input listeners
        logger.info("初始化输入监听器...")
        try:
            self.setupInputListeners()
            logger.info("✓ 输入监听器初始化成功")
        except Exception as e:
            logger.error(f"✗ 输入监听器初始化失败: {e}")
            logger.error("请安装 pynput: pip install pynput")
            raise
        
    def setupInputListeners(self):
        """Setup keyboard and mouse listeners"""
        
        # Keyboard listener
        self.keyboard_listener_ = keyboard.Listener(
            on_press=self.onKeyPress,
            on_release=self.onKeyRelease
        )
        
        # Mouse listener
        self.mouse_listener_ = mouse.Listener(
            on_move=self.onMouseMove,
            on_click=self.onMouseClick
        )
        
        self.keyboard_listener_.start()
        self.mouse_listener_.start()
        
    def onKeyPress(self, key):
        """Keyboard press callback"""
        try:
            self.current_keys_.add(key.char)
        except AttributeError:
            # Special key
            self.current_keys_.add(str(key))
            
    def onKeyRelease(self, key):
        """Keyboard release callback"""
        try:
            self.current_keys_.discard(key.char)
        except AttributeError:
            self.current_keys_.discard(str(key))
            
        # Stop recording on ESC - 只设置标志，让主循环处理
        if key == keyboard.Key.esc:
            logger.info("检测到 ESC 键，准备停止录制...")
            self.recording_ = False
            
    def onMouseMove(self, x, y):
        """Mouse move callback"""
        self.mouse_position_ = (x, y)
        
    def onMouseClick(self, x, y, button, pressed):
        """Mouse click callback"""
        if pressed:
            self.current_keys_.add(f"mouse_{button.name}")
        else:
            self.current_keys_.discard(f"mouse_{button.name}")
            
    def captureScreen(self) -> np.ndarray:
        """Capture current screen"""
        try:
            if self.capture_mode_ == "window":
                # 窗口捕获模式
                logger.debug("Using window capture")
                if not self.window_capture_ or not self.window_capture_.IsValid():
                    # 尝试重新查找窗口
                    if self.window_capture_:
                        self.window_capture_.findWindow()
                    
                    if not self.window_capture_ or not self.window_capture_.IsValid():
                        raise RuntimeError("窗口已关闭或无效")
                
                frame = self.window_capture_.Capture()
                if frame is None:
                    raise RuntimeError("窗口捕获返回 None")
                    
            elif self.capture_mode_ == "fullscreen_cpp":
                # C++ 全屏捕获
                logger.debug("Using C++ screen capture")
                buffer = self.screen_capture_.Capture()
                
                # pybind11 可能返回 list 或 bytes
                if isinstance(buffer, list):
                    frame = np.array(buffer, dtype=np.uint8)
                else:
                    frame = np.frombuffer(buffer, dtype=np.uint8)
                
                frame = frame.reshape((1080, 1920, 3))
                
            elif self.capture_mode_ == "fullscreen_mss":
                # Python fallback 全屏捕获
                logger.debug("Using Python fallback (mss)")
                screenshot = self.sct_.grab(self.monitor_)
                frame = np.array(screenshot)[:, :, :3]  # Remove alpha channel
            else:
                raise ValueError(f"Unknown capture mode: {self.capture_mode_}")
            
            logger.debug(f"Captured frame: shape={frame.shape}, dtype={frame.dtype}")
            return frame
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def getCurrentAction(self) -> dict:
        """Get current input action"""
        return {
            "keys": list(self.current_keys_),
            "mouse_pos": self.mouse_position_
        }
        
    def startRecording(self):
        """Start recording"""
        self.recording_ = True
        self.saved_ = False  # 重置保存标志
        self.frames_ = []  # 仅视频模式使用
        self.actions_ = []
        self.frame_numbers_ = []
        
        # 创建 session 目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir_ = self.output_dir_ / f"session_{timestamp}"
        self.session_dir_.mkdir(parents=True, exist_ok=True)
        
        # 如果保存帧，创建 frames 目录并启动异步保存器
        if self.save_format_ in ["frames", "both"]:
            self.frames_dir_ = self.session_dir_ / "frames"
            self.frames_dir_.mkdir(exist_ok=True)
            
            # 启动异步帧保存器（如果可用）
            if ASYNC_SAVER_AVAILABLE:
                self.async_saver_ = AsyncFrameSaver(
                    frames_dir=self.frames_dir_,
                    jpeg_quality=95,
                    queue_size=60  # 队列可容纳 2 秒的帧（30 FPS）
                )
                self.async_saver_.Start()
        
        logger.info("=" * 80)
        logger.info("🎬 录制开始！")
        logger.info("  - 按 ESC 键停止录制")
        logger.info(f"  - 目标 FPS: {self.fps_}")
        logger.info(f"  - 捕获模式: {self.capture_mode_}")
        logger.info(f"  - 保存格式: {self.save_format_}")
        logger.info(f"  - 帧采样: 每 {self.frame_step_} 帧保存一次")
        logger.info(f"  - 会话目录: {self.session_dir_}")
        if self.capture_mode_ == "window":
            logger.info(f"  - 窗口大小: {self.window_capture_.GetWidth()}x{self.window_capture_.GetHeight()}")
        logger.info("=" * 80)
        
        start_time = time.time()
        frame_count = 0
        last_log_time = start_time
        
        try:
            while self.recording_:
                frame_start = time.time()
                
                try:
                    # Capture frame
                    frame = self.captureScreen()
                    action = self.getCurrentAction()
                    timestamp = time.time() - start_time
                    
                    # 调试：保存第一帧为图片
                    if frame_count == 0:
                        debug_path = self.session_dir_ / "debug_first_frame.png"
                        cv2.imwrite(str(debug_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        logger.info(f"[调试] 第一帧已保存到: {debug_path}")
                        logger.info(f"[调试] 帧信息: shape={frame.shape}, mean={frame.mean():.2f}, max={frame.max()}, min={frame.min()}")
                    
                    # 实时保存帧（如果启用）
                    if self.save_format_ in ["frames", "both"]:
                        if frame_count % self.frame_step_ == 0:
                            # 使用异步保存器（如果可用）
                            if self.async_saver_:
                                self.async_saver_.SaveFrame(frame, frame_count)
                            else:
                                # 回退到同步保存
                                frame_path = self.frames_dir_ / f"frame_{frame_count:06d}.jpg"
                                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # 仅在视频模式下存储到内存
                    if self.save_format_ in ["video", "both"]:
                        self.frames_.append(frame)
                    
                    # 记录操作和对应的帧号
                    self.actions_.append(action)
                    self.frame_numbers_.append(frame_count)
                    
                    frame_count += 1
                    
                    # Log progress every 5 seconds
                    if time.time() - last_log_time >= 5.0:
                        actual_fps = frame_count / timestamp if timestamp > 0 else 0
                        log_msg = f"录制中... 帧数: {frame_count}, 时长: {timestamp:.1f}s, FPS: {actual_fps:.1f}"
                        
                        # 显示异步保存队列状态
                        if self.async_saver_:
                            queue_size = self.async_saver_.GetQueueSize()
                            log_msg += f", 队列: {queue_size}/60"
                        
                        logger.info(log_msg)
                        last_log_time = time.time()
                        
                except Exception as e:
                    logger.error(f"捕获帧失败: {e}")
                    # Continue trying instead of crashing
                    
                # Maintain FPS
                elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_interval_ - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("录制被用户中断")
        except Exception as e:
            logger.error(f"录制过程出错: {e}")
            import traceback
            traceback.print_exc()
            
        self.stopRecording()
        
    def stopRecording(self):
        """Stop recording and save data"""
        # 使用 saved_ 标志避免重复保存
        if self.saved_:
            logger.debug("数据已经保存过，跳过")
            return
            
        self.recording_ = False  # 确保停止录制
        self.saved_ = True  # 标记为已保存（即使失败也不重试）
        
        # 停止异步帧保存器（等待队列清空）
        if self.async_saver_:
            logger.info("等待异步保存队列清空...")
            self.async_saver_.Stop()
            stats = self.async_saver_.GetStats()
            logger.info(f"异步保存统计: 已保存={stats['saved']}, 丢弃={stats['dropped']}")
        
        logger.info("Stopping recording...")
        logger.info(f"录制完成，共 {len(self.frame_numbers_)} 帧")
        
        # Check if any data was recorded
        if not self.frame_numbers_ or not self.actions_:
            logger.error("=" * 80)
            logger.error("❌ 录制失败：没有捕获到任何帧！")
            logger.error("=" * 80)
            logger.error("可能的原因：")
            logger.error("  1. 录制时间太短（立即按了 ESC）")
            logger.error("  2. 屏幕捕获模块初始化失败")
            logger.error("  3. C++ 绑定未编译或加载失败")
            logger.error("  4. Python fallback (mss) 未安装或无法访问屏幕")
            logger.error("=" * 80)
            logger.error(f"调试信息：")
            logger.error(f"  - 捕获的帧数: {len(self.frame_numbers_)}")
            logger.error(f"  - 操作记录数: {len(self.actions_)}")
            logger.error(f"  - C++ 可用: {CPP_AVAILABLE}")
            logger.error("=" * 80)
            return
        
        try:
            # 保存元数据
            logger.info("保存会话元数据...")
            total_frames = len(self.frame_numbers_)
            duration = (self.frame_numbers_[-1] / self.fps_) if total_frames > 0 else 0
            
            # 获取分辨率信息
            if self.save_format_ in ["frames", "both"]:
                # 从已保存的第一帧读取分辨率
                first_frame_path = self.frames_dir_ / "frame_000000.jpg"
                if first_frame_path.exists():
                    first_frame = cv2.imread(str(first_frame_path))
                    resolution = [first_frame.shape[1], first_frame.shape[0]]
                else:
                    resolution = [1920, 1080]  # 默认值
            elif self.save_format_ == "video" and self.frames_:
                resolution = [self.frames_[0].shape[1], self.frames_[0].shape[0]]
            else:
                resolution = [1920, 1080]
            
            meta_data = {
                "session_id": self.session_dir_.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fps": self.fps_,
                "total_frames": total_frames,
                "duration_seconds": duration,
                "resolution": resolution,
                "capture_mode": self.capture_mode_,
                "save_format": self.save_format_,
                "frame_step": self.frame_step_,
                "frames_saved": total_frames // self.frame_step_ if self.save_format_ in ["frames", "both"] else 0
            }
            
            meta_path = self.session_dir_ / "meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ 元数据已保存: {meta_path}")
            
            # 保存操作记录（关联到帧号）
            logger.info("保存操作记录...")
            actions_data = []
            for i, (frame_num, action) in enumerate(zip(self.frame_numbers_, self.actions_)):
                actions_data.append({
                    "frame": frame_num,
                    "timestamp": frame_num / self.fps_,
                    "keys": action["keys"],
                    "mouse_pos": action["mouse_pos"]
                })
            
            actions_path = self.session_dir_ / "actions.json"
            with open(actions_path, 'w', encoding='utf-8') as f:
                json.dump(actions_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ 操作记录已保存: {actions_path} ({len(actions_data)} 条)")
            
            # 如果需要保存视频（可选）
            if self.save_format_ in ["video", "both"] and self.frames_:
                logger.info("正在编码视频...")
                self._saveVideo()
            
            # Summary
            logger.info("=" * 80)
            logger.info(f"✓ 录制成功保存到: {self.session_dir_}")
            logger.info(f"  - 总帧数: {total_frames}")
            logger.info(f"  - 时长: {duration:.2f}s")
            logger.info(f"  - 平均 FPS: {total_frames / duration:.2f}" if duration > 0 else "  - 平均 FPS: N/A")
            if self.save_format_ in ["frames", "both"]:
                saved_frames = total_frames // self.frame_step_
                logger.info(f"  - 已保存帧: {saved_frames} 张 JPEG")
            if self.save_format_ in ["video", "both"]:
                logger.info(f"  - 视频: gameplay.avi")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ 保存录制数据时出错: {e}")
            logger.error("=" * 80)
            import traceback
            traceback.print_exc()
    
    def _saveVideo(self):
        """保存视频文件（仅在 save_format 包含 'video' 时调用）"""
        if not self.frames_:
            logger.warning("没有帧数据，跳过视频保存")
            return
        
        frame_height, frame_width = self.frames_[0].shape[:2]
        video_path = self.session_dir_ / "gameplay.avi"
        
        # 使用 XVID 编解码器（最兼容）
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.fps_,
            (frame_width, frame_height)
        )
        
        if not out.isOpened():
            logger.error("无法打开视频写入器")
            return
        
        for i, frame in enumerate(self.frames_):
            if i % 100 == 0:
                logger.info(f"编码帧 {i}/{len(self.frames_)}...")
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        
        if video_path.exists() and video_path.stat().st_size > 1024:
            logger.info(f"✓ 视频已保存: {video_path} ({video_path.stat().st_size:,} bytes)")
        else:
            logger.error("视频文件创建失败或为空")
        
    def cleanup(self):
        """Clean up resources"""
        self.keyboard_listener_.stop()
        self.mouse_listener_.stop()


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Record World of Tanks gameplay")
    parser.add_argument(
        "--output",
        type=str,
        default="data/recordings",
        help="Output directory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording FPS"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["window", "fullscreen"],
        default="window",
        help="Capture mode: 'window' (recommended) or 'fullscreen'"
    )
    parser.add_argument(
        "--process",
        type=str,
        default="WorldOfTanks.exe",
        help="Process name for window capture (e.g., 'WorldOfTanks.exe')"
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default=None,
        help="Window title for window capture (alternative to --process)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test screen capture without recording"
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["frames", "video", "both"],
        default="frames",
        help="保存格式: 'frames' (仅帧), 'video' (仅视频), 'both' (两者都有)"
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="帧采样间隔 (1=保存所有帧, 5=每5帧保存一次)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🎮 World of Tanks - 游戏录制工具")
    logger.info("=" * 80)
    logger.info(f"输出目录: {args.output}")
    logger.info(f"录制 FPS: {args.fps}")
    logger.info(f"捕获模式: {args.mode}")
    logger.info(f"保存格式: {args.save_format}")
    logger.info(f"帧采样: 每 {args.frame_step} 帧保存一次")
    if args.mode == "window":
        if args.process:
            logger.info(f"目标进程: {args.process}")
        elif args.window_title:
            logger.info(f"目标窗口: {args.window_title}")
        logger.info(f"窗口捕获: {'可用' if WINDOW_CAPTURE_AVAILABLE else '不可用'}")
    logger.info(f"C++ 加速: {'可用' if CPP_AVAILABLE else '不可用 (将使用 Python fallback)'}")
    logger.info("")
    logger.info("使用说明：")
    logger.info("  1. 启动《坦克世界》并进入战斗")
    logger.info("  2. 按 Enter 开始录制")
    logger.info("  3. 正常游戏")
    logger.info("  4. 按 ESC 键停止录制并保存")
    logger.info("")
    logger.info("注意事项：")
    logger.info("  - 请至少录制 3-5 秒以确保有数据保存")
    if args.save_format == "frames":
        logger.info("  - frames 模式内存占用小，推荐长时间录制")
    elif args.save_format == "video":
        logger.info("  - video 模式会占用内存，时间过长可能导致内存不足")
    else:
        logger.info("  - both 模式会同时保存帧和视频，占用较多内存")
    if args.mode == "window":
        logger.info("  - window 模式会自动跟踪游戏窗口（推荐）")
    else:
        logger.info("  - fullscreen 模式捕获整个屏幕")
    logger.info("=" * 80)
    
    if args.test:
        logger.info("\n[测试模式] 测试屏幕捕获功能...")
        try:
            if CPP_AVAILABLE:
                sc = ScreenCapture(1920, 1080)
                logger.info("✓ C++ ScreenCapture 初始化成功")
                buffer = sc.Capture()
                logger.info(f"✓ 捕获成功: {len(buffer)} 字节")
                logger.info(f"✓ 预期大小: {1920 * 1080 * 3} 字节")
            else:
                import mss
                sct = mss.mss()
                logger.info("✓ mss 初始化成功")
                monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)[:, :, :3]
                logger.info(f"✓ 捕获成功: shape={frame.shape}, dtype={frame.dtype}")
            
            logger.info("\n✓ 测试通过！屏幕捕获功能正常。")
            logger.info("可以开始正式录制了。")
        except Exception as e:
            logger.error(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        return
    
    input("\n按 Enter 键继续...")
    
    # Create recorder
    try:
        logger.info("\n正在初始化录制器...")
        recorder = GameplayRecorder(
            output_dir=args.output,
            fps=args.fps,
            capture_mode=args.mode,
            process_name=args.process if args.mode == "window" else None,
            window_title=args.window_title if args.mode == "window" else None,
            save_format=args.save_format,
            frame_step=args.frame_step
        )
        logger.info("✓ 录制器初始化成功\n")
    except Exception as e:
        logger.error(f"\n✗ 录制器初始化失败: {e}")
        logger.error("\n故障排查：")
        logger.error("  1. 检查是否安装了依赖: pip install pywin32 psutil mss pynput opencv-python")
        logger.error("  2. 检查 C++ 模块是否正确编译")
        logger.error("  3. 确保游戏已启动（window 模式）")
        logger.error("  4. 确保有屏幕访问权限")
        if args.mode == "window":
            logger.error(f"  5. 检查进程名是否正确: {args.process or args.window_title}")
            logger.error("  6. 尝试使用 fullscreen 模式: --mode fullscreen")
        return
    
    try:
        recorder.startRecording()
    except Exception as e:
        logger.error(f"录制过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.cleanup()
        logger.info("\n录制器已清理")


if __name__ == "__main__":
    main()

