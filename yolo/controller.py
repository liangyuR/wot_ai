#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主控制器：协调所有模块，实现主循环逻辑
"""

from pathlib import Path
from typing import Optional, Callable
import time
import logging
import threading

from core.screen_capture import ScreenCapture, MssScreenCapture
from core.detection_engine import DetectionEngine
from core.aim_controller import AimController
from core.hotkey_manager import HotkeyManager
from core.config_manager import AimConfigManager
from aim_assist import AimAssist

logger = logging.getLogger(__name__)


class AimAssistMainController:
    """瞄准辅助主控制器"""
    
    def __init__(self, config_path: Path):
        """
        初始化主控制器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path_ = config_path
        self.config_manager_ = AimConfigManager(config_path)
        
        # 模块
        self.screen_capture_: Optional[ScreenCapture] = None
        self.detection_engine_: Optional[DetectionEngine] = None
        self.aim_controller_: Optional[AimController] = None
        self.hotkey_manager_: Optional[HotkeyManager] = None
        self.aim_assist_: Optional[AimAssist] = None
        
        # 状态
        self.is_running_ = False
        self.is_enabled_ = False
        self.control_thread_: Optional[threading.Thread] = None
        self.stop_flag_ = threading.Event()
        
        # 回调
        self.status_callback_: Optional[Callable[[str], None]] = None
        self.stats_callback_: Optional[Callable[[dict], None]] = None
        
        # 统计信息
        self.stats_ = {
            'frame_count': 0,
            'detection_count': 0,
            'fps': 0.0,
            'last_update_time': time.time()
        }
    
    def Initialize(self) -> bool:
        """
        初始化所有模块
        
        Returns:
            是否初始化成功
        """
        try:
            # 加载配置
            config = self.config_manager_.Load()
            
            # 验证配置
            is_valid, errors = self.config_manager_.Validate()
            if not is_valid:
                logger.error(f"配置验证失败: {errors}")
                return False
            
            # 初始化屏幕捕获
            self.screen_capture_ = MssScreenCapture()
            
            # 初始化检测引擎
            model_path = self.config_manager_.Get('detection.model_path', 'train/model/yolo11n.pt')
            self.detection_engine_ = DetectionEngine(model_path, base_dir=self.config_path_.parent)
            if not self.detection_engine_.LoadModel():
                logger.error("检测引擎初始化失败")
                return False
            
            # 初始化瞄准辅助
            screen_w = self.config_manager_.Get('screen.width', 1920)
            screen_h = self.config_manager_.Get('screen.height', 1080)
            h_fov = self.config_manager_.Get('fov.horizontal', 90.0)
            v_fov = self.config_manager_.Get('fov.vertical', None)
            mouse_sensitivity = self.config_manager_.Get('mouse.sensitivity', 1.0)
            calibration_factor = self.config_manager_.Get('mouse.calibration_factor', None)
            
            self.aim_assist_ = AimAssist(
                screen_width=screen_w,
                screen_height=screen_h,
                horizontal_fov=h_fov,
                vertical_fov=v_fov,
                mouse_sensitivity=mouse_sensitivity,
                calibration_factor=calibration_factor
            )
            
            # 初始化瞄准控制器
            smoothing_factor = self.config_manager_.Get('smoothing.factor', 0.3)
            max_step = self.config_manager_.Get('smoothing.max_step', 50.0)
            self.aim_controller_ = AimController(
                self.aim_assist_,
                smoothing_factor=smoothing_factor,
                max_step=max_step
            )
            
            # 初始化热键管理器
            self.hotkey_manager_ = HotkeyManager()
            toggle_key = self.config_manager_.Get('hotkeys.toggle', 'f8')
            exit_key = self.config_manager_.Get('hotkeys.exit', 'esc')
            
            self.hotkey_manager_.RegisterHotkey('toggle', toggle_key, self.ToggleEnabled)
            self.hotkey_manager_.RegisterHotkey('exit', exit_key, self.Stop)
            
            logger.info("所有模块初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def ToggleEnabled(self):
        """切换启用/禁用状态"""
        self.is_enabled_ = not self.is_enabled_
        status = "启用" if self.is_enabled_ else "禁用"
        logger.info(f"瞄准辅助: {status}")
        
        if not self.is_enabled_:
            # 重置平滑状态
            if self.aim_controller_:
                self.aim_controller_.ResetSmoothing()
        
        if self.status_callback_:
            self.status_callback_(status)
    
    def Start(self):
        """启动主循环"""
        if self.is_running_:
            logger.warning("控制器已在运行")
            return
        
        if not self.Initialize():
            logger.error("初始化失败，无法启动")
            return
        
        # 启动热键监听
        self.hotkey_manager_.Start()
        
        # 启动控制线程
        self.stop_flag_.clear()
        self.is_running_ = True
        self.control_thread_ = threading.Thread(target=self._ControlLoop, daemon=True)
        self.control_thread_.start()
        
        logger.info("主控制器已启动")
        if self.status_callback_:
            self.status_callback_("运行中")
    
    def Stop(self):
        """停止主循环"""
        if not self.is_running_:
            return
        
        self.is_enabled_ = False
        self.stop_flag_.set()
        self.is_running_ = False
        
        # 等待线程结束
        if self.control_thread_ and self.control_thread_.is_alive():
            self.control_thread_.join(timeout=2.0)
        
        # 停止热键监听
        if self.hotkey_manager_:
            self.hotkey_manager_.Stop()
        
        logger.info("主控制器已停止")
        if self.status_callback_:
            self.status_callback_("已停止")
    
    def _ControlLoop(self):
        """主控制循环"""
        fps = self.config_manager_.Get('detection.fps', 30.0)
        interval = 1.0 / fps
        target_class = self.config_manager_.Get('detection.target_class', None)
        
        last_log_time = time.time()
        frame_times = []
        
        logger.info("控制循环开始")
        
        try:
            while not self.stop_flag_.is_set():
                frame_start_time = time.time()
                
                # 捕获屏幕
                frame = self.screen_capture_.Capture()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # 执行检测
                detections = self.detection_engine_.Detect(frame)
                self.stats_['detection_count'] = len(detections)
                
                # 如果启用且有检测结果，处理目标
                if self.is_enabled_ and detections:
                    target_pos = self.detection_engine_.GetBestTarget(detections, target_class)
                    
                    if target_pos and self.aim_controller_:
                        target_x, target_y = target_pos
                        try:
                            self.aim_controller_.ProcessTarget(target_x, target_y)
                        except Exception as e:
                            logger.error(f"处理目标失败: {e}")
                
                # 更新统计信息
                self.stats_['frame_count'] += 1
                frame_times.append(time.time() - frame_start_time)
                
                # 计算 FPS（最近10帧的平均值）
                if len(frame_times) > 10:
                    frame_times.pop(0)
                if frame_times:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    self.stats_['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                
                # 定期日志和回调
                current_time = time.time()
                if current_time - last_log_time >= 2.0:
                    logger.debug(f"FPS: {self.stats_['fps']:.1f}, "
                               f"帧数: {self.stats_['frame_count']}, "
                               f"检测: {self.stats_['detection_count']}, "
                               f"状态: {'启用' if self.is_enabled_ else '禁用'}")
                    last_log_time = current_time
                    
                    if self.stats_callback_:
                        self.stats_callback_(self.stats_.copy())
                
                # 维持 FPS
                elapsed = time.time() - frame_start_time
                sleep_time = max(0.0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"控制循环异常: {e}")
        finally:
            logger.info("控制循环结束")
    
    def SetStatusCallback(self, callback: Callable[[str], None]):
        """设置状态回调"""
        self.status_callback_ = callback
    
    def SetStatsCallback(self, callback: Callable[[dict], None]):
        """设置统计信息回调"""
        self.stats_callback_ = callback
    
    def UpdateConfig(self, config_updates: dict):
        """
        更新配置（热更新）
        
        Args:
            config_updates: 配置更新字典（键路径 -> 值）
        """
        for key_path, value in config_updates.items():
            self.config_manager_.Set(key_path, value)
        
        # 重新初始化相关模块
        if self.aim_assist_ and self.aim_controller_:
            # 重新创建 AimAssist
            screen_w = self.config_manager_.Get('screen.width', 1920)
            screen_h = self.config_manager_.Get('screen.height', 1080)
            h_fov = self.config_manager_.Get('fov.horizontal', 90.0)
            v_fov = self.config_manager_.Get('fov.vertical', None)
            mouse_sensitivity = self.config_manager_.Get('mouse.sensitivity', 1.0)
            calibration_factor = self.config_manager_.Get('mouse.calibration_factor', None)
            
            new_aim_assist = AimAssist(
                screen_width=screen_w,
                screen_height=screen_h,
                horizontal_fov=h_fov,
                vertical_fov=v_fov,
                mouse_sensitivity=mouse_sensitivity,
                calibration_factor=calibration_factor
            )
            
            # 更新平滑配置
            smoothing_factor = self.config_manager_.Get('smoothing.factor', 0.3)
            max_step = self.config_manager_.Get('smoothing.max_step', 50.0)
            self.aim_controller_.UpdateSmoothingConfig(smoothing_factor, max_step)
            self.aim_controller_.UpdateAimAssist(new_aim_assist)
            
            logger.info("配置已热更新")
    
    def GetStats(self) -> dict:
        """获取统计信息"""
        return self.stats_.copy()
    
    def IsRunning(self) -> bool:
        """是否正在运行"""
        return self.is_running_
    
    def IsEnabled(self) -> bool:
        """是否启用"""
        return self.is_enabled_

