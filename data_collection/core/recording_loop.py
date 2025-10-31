"""
录制循环核心逻辑
"""

import time
import cv2
import numpy as np
from typing import Optional, Callable
from pathlib import Path
from loguru import logger

from data_collection.core.screen_capture import ScreenCapture
from data_collection.core.input_listener import InputListener, InputAction


class RecordingLoop:
    """录制循环，负责帧捕获和保存的核心逻辑"""
    
    def __init__(
        self,
        screen_capture: ScreenCapture,
        input_listener: InputListener,
        output_dir: Path,
        fps: int = 30,
        frame_step: int = 1,
        save_format: str = "frames",
        frame_saver: Optional[Callable] = None,
        on_frame_captured: Optional[Callable] = None
    ):
        """
        初始化录制循环
        
        Args:
            screen_capture: 屏幕捕获器
            input_listener: 输入监听器
            output_dir: 输出目录
            fps: 录制帧率
            frame_step: 帧采样间隔（每 N 帧保存一次）
            save_format: 保存格式 'frames', 'video', 'both'
            frame_saver: 帧保存回调函数 callback(frame, frame_number) -> bool
            on_frame_captured: 帧捕获回调函数 callback(frame, frame_number, timestamp)
        """
        self.screen_capture_ = screen_capture
        self.input_listener_ = input_listener
        self.output_dir_ = Path(output_dir)
        self.output_dir_.mkdir(parents=True, exist_ok=True)
        
        self.fps_ = fps
        self.frame_interval_ = 1.0 / fps
        self.frame_step_ = frame_step
        self.save_format_ = save_format
        
        self.frame_saver_ = frame_saver
        self.on_frame_captured_ = on_frame_captured
        
        # 录制状态
        self.is_recording_ = False
        self.frame_count_ = 0
        self.start_time_ = 0.0
        
        # 数据存储
        self.actions_ = []
        self.frame_numbers_ = []
        self.frames_ = []  # 仅 video 模式使用
        
        # 会话目录
        self.session_dir_ = None
        self.frames_dir_ = None
    
    def PrepareSession(self):
        """准备录制会话（创建目录等）"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir_ = self.output_dir_ / f"session_{timestamp}"
        self.session_dir_.mkdir(parents=True, exist_ok=True)
        
        if self.save_format_ in ["frames", "both"]:
            self.frames_dir_ = self.session_dir_ / "frames"
            self.frames_dir_.mkdir(exist_ok=True)
        
        # 重置状态
        self.frame_count_ = 0
        self.actions_ = []
        self.frame_numbers_ = []
        self.frames_ = []
        self.start_time_ = time.time()
        
        logger.info(f"录制会话已准备: {self.session_dir_}")
    
    def Start(self):
        """开始录制"""
        self.is_recording_ = True
        self.PrepareSession()
    
    def Stop(self):
        """停止录制"""
        self.is_recording_ = False
    
    def IsRecording(self) -> bool:
        """是否正在录制"""
        return self.is_recording_
    
    def RunFrame(self) -> Optional[np.ndarray]:
        """
        运行一帧（捕获、处理、保存）
        
        Returns:
            捕获的帧，失败返回 None
        """
        if not self.is_recording_:
            return None
        
        frame_start_time = time.time()
        
        try:
            # 捕获屏幕
            frame = self.screen_capture_.Capture()
            if frame is None:
                logger.debug("屏幕捕获失败")
                return None
            
            # 获取输入
            action = InputAction()
            action.keys_ = self.input_listener_.GetPressedKeys()
            action.mouse_pos_ = self.input_listener_.GetMousePosition()
            
            timestamp = time.time() - self.start_time_
            
            # 保存第一帧用于调试
            if self.frame_count_ == 0:
                debug_path = self.session_dir_ / "debug_first_frame.png"
                cv2.imwrite(str(debug_path), frame)
                logger.info(f"[调试] 第一帧已保存: {debug_path.name}")
            
            # 保存帧（如果需要）
            if self.save_format_ in ["frames", "both"]:
                if self.frame_count_ % self.frame_step_ == 0:
                    if self.frame_saver_:
                        self.frame_saver_(frame, self.frame_count_)
            
            # 视频模式存储到内存
            if self.save_format_ in ["video", "both"]:
                self.frames_.append(frame.copy())  # 需要复制，因为 frame 可能被重用
            
            # 记录操作
            self.actions_.append(action.ToDict())
            self.frame_numbers_.append(self.frame_count_)
            
            # 回调
            if self.on_frame_captured_:
                self.on_frame_captured_(frame, self.frame_count_, timestamp)
            
            self.frame_count_ += 1
            
            # 维持 FPS
            elapsed = time.time() - frame_start_time
            sleep_time = max(0, self.frame_interval_ - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            return frame
            
        except Exception as e:
            logger.error(f"处理帧失败: {e}")
            return None
    
    def GetStats(self) -> dict:
        """获取统计信息"""
        return {
            'frame_count': self.frame_count_,
            'duration': time.time() - self.start_time_ if self.start_time_ > 0 else 0.0,
            'actions_count': len(self.actions_)
        }
    
    def GetActions(self) -> list:
        """获取操作记录"""
        return self.actions_
    
    def GetFrameNumbers(self) -> list:
        """获取帧号列表"""
        return self.frame_numbers_
    
    def GetFrames(self) -> list:
        """获取帧列表（仅 video 模式）"""
        return self.frames_
    
    def GetSessionDir(self) -> Optional[Path]:
        """获取会话目录"""
        return self.session_dir_
    
    def GetFramesDir(self) -> Optional[Path]:
        """获取帧目录"""
        return self.frames_dir_

