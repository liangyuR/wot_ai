#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航主程序

封装导航系统的核心逻辑，负责初始化、运行和停止导航功能。
"""

import time
import queue
import threading
from pathlib import Path
from typing import Optional
import signal

from loguru import logger

# 导入所需模块
from wot_ai.game_modules.navigation.service.capture_service import CaptureService
from wot_ai.game_modules.vision.detection.minimap_anchor_detector import MinimapAnchorDetector
from wot_ai.game_modules.vision.detection.minimap_detector import MinimapDetector
from wot_ai.game_modules.navigation.service.control_service import ControlService
from wot_ai.game_modules.navigation.core.navigation_executor import NavigationExecutor

# 导入新模块
from wot_ai.game_modules.navigation.core.mask_loader import load_mask
from wot_ai.game_modules.navigation.service.minimap_service import MinimapService
from wot_ai.game_modules.navigation.service.path_planning_service import PathPlanningService
from wot_ai.game_modules.navigation.service.thread_manager import ThreadManager
from wot_ai.game_modules.navigation.config.models import NavigationConfig


class NavigationMain:
    """导航主程序"""
    
    def __init__(self, config: NavigationConfig, map_name: Optional[str] = None):
        """
        初始化导航主程序
        
        Args:
            config: NavigationConfig配置对象
            map_name: 当前地图名称（可选）
        """
        self.config_ = config
        self.map_name_ = map_name
        self.running_ = False
        
        # 服务实例
        self.capture_service_: Optional[CaptureService] = None
        self.anchor_detector_: Optional[MinimapAnchorDetector] = None
        self.minimap_detector_: Optional[MinimapDetector] = None
        self.control_service_: Optional[ControlService] = None
        self.nav_executor_: Optional[NavigationExecutor] = None
        
        # 新服务实例
        self.minimap_service_: Optional[MinimapService] = None
        self.path_planning_service_: Optional[PathPlanningService] = None
        self.thread_manager_: Optional[ThreadManager] = None
        
        # 掩码数据
        self.mask_data_ = None
        
        # 使用队列传递数据（完全无锁）
        self.detection_queue_ = queue.Queue(maxsize=10)  # 支持30FPS
        self.path_queue_ = queue.Queue(maxsize=1)  # 只保留最新路径
        
        # 注册信号处理（仅在主线程中）
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._SignalHandler_)
                signal.signal(signal.SIGTERM, self._SignalHandler_)
            except ValueError:
                # 如果信号注册失败（例如在非主线程），记录警告但不中断
                logger.warning("无法注册信号处理器（非主线程），将使用其他方式处理退出")
        else:
            logger.debug("非主线程，跳过信号注册")
    
    def set_map_name(self, map_name: Optional[str]) -> None:
        """更新地图名称"""
        self.map_name_ = map_name
    
    def _SignalHandler_(self, signum, frame):
        """信号处理器（Ctrl+C）"""
        logger.info("收到退出信号，正在关闭...")
        self.Stop()
    
    def Initialize(self) -> bool:
        """
        初始化所有服务
        
        Returns:
            是否初始化成功
        """
        try:
            logger.info("开始初始化服务...")
            
            # 1. 初始化屏幕捕获服务
            logger.info("初始化屏幕捕获服务...")
            self.capture_service_ = CaptureService(monitor_index=self.config_.monitor_index)
            
            # 2. 初始化小地图锚点检测器
            logger.info(f"初始化小地图锚点检测器: {self.config_.minimap.template_path}")
            self.anchor_detector_ = MinimapAnchorDetector(
                template_path=self.config_.minimap.template_path,
                debug=False,
                multi_scale=False
            )
            
            # 3. 初始化YOLO检测器
            logger.info(f"初始化YOLO检测器: {self.config_.model.path}")
            self.minimap_detector_ = MinimapDetector(
                model_path=self.config_.model.path,
                conf_threshold=self.config_.model.conf_threshold,
                iou_threshold=self.config_.model.iou_threshold,
            )
            
            if not self.minimap_detector_.LoadModel():
                logger.error("YOLO模型加载失败")
                return False
            
            # 模型预热：避免首帧延迟
            logger.info(f"模型预热中，使用尺寸: {self.config_.minimap.size}")
            self.minimap_detector_.engine_.Warmup(img_size=self.config_.minimap.size)
            
            # 4. 初始化控制服务
            logger.info("初始化控制服务...")
            self.control_service_ = ControlService()
            
            # 5. 初始化导航执行器
            logger.info("初始化导航执行器...")
            self.nav_executor_ = NavigationExecutor(
                control_service=self.control_service_,
                move_speed=self.config_.control.move_speed,
                rotation_smooth=self.config_.control.rotation_smooth
            )
            
            # 6. 初始化小地图服务
            logger.info("初始化小地图服务...")
            self.minimap_service_ = MinimapService(self.anchor_detector_, self.config_)
            
            # 7. 初始化路径规划服务
            logger.info("初始化路径规划服务...")
            self.path_planning_service_ = PathPlanningService(self.config_)
            
            logger.info("所有服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def Run(self):
        """主循环：启动多线程架构"""
        logger.info("启动导航主循环（多线程模式）...")
        self.running_ = True
        
        # 首次检测小地图位置
        logger.info("首次检测小地图位置...")
        frame = self.capture_service_.Capture()
        if frame is None:
            logger.error("无法捕获屏幕")
            return
        
        minimap_region = self.minimap_service_.detect_region(frame)
        if minimap_region is None:
            logger.error("无法检测到小地图位置，退出")
            return
        
        # 加载掩码（在初始化时调用一次）
        minimap_size = (minimap_region['width'], minimap_region['height'])
        mask_path = self._ResolveMaskPath_()
        
        self.mask_data_ = load_mask(mask_path, minimap_size, self.config_.grid.size, self.config_)
        if self.mask_data_ is None:
            logger.error("掩码加载失败，退出")
            return
        
        # 设置路径规划服务的掩码数据
        self.path_planning_service_.set_mask_data(
            self.mask_data_.grid,
            self.mask_data_.cost_map,
            self.mask_data_.inflated_obstacle
        )
        
        # 初始化透明覆盖层
        if not self.minimap_service_.initialize_overlay(minimap_region):
            logger.error("透明覆盖层初始化失败，退出")
            return
        
        # 创建线程管理器
        queues = {
            'detection': self.detection_queue_,
            'path': self.path_queue_
        }
        
        self.thread_manager_ = ThreadManager(
            capture_service=self.capture_service_,
            minimap_detector=self.minimap_detector_,
            minimap_service=self.minimap_service_,
            path_planning_service=self.path_planning_service_,
            nav_executor=self.nav_executor_,
            queues=queues,
            mask_data=self.mask_data_,
            config=self.config_
        )
        
        # 启动所有线程
        if not self.thread_manager_.start_all():
            logger.error("线程管理器启动失败")
            return
        
        logger.info("所有线程已启动，等待线程运行...")
        
        # 主线程等待所有线程完成
        try:
            while self.running_ and self.thread_manager_.is_running():
                time.sleep(5)
                # 检查线程是否还在运行
                if self.thread_manager_.detection_thread_ and not self.thread_manager_.detection_thread_.is_alive():
                    logger.warning("检测线程已退出")
                if self.thread_manager_.control_thread_ and not self.thread_manager_.control_thread_.is_alive():
                    logger.warning("控制线程已退出")
                if self.thread_manager_.ui_thread_ and not self.thread_manager_.ui_thread_.is_alive():
                    logger.warning("UI更新线程已退出")
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号")
        finally:
            logger.info("正在停止所有线程...")
            self.Stop()
    
    def _ResolveMaskPath_(self) -> Optional[Path]:
        """根据配置与地图名称解析掩码路径"""
        if not self.config_.mask:
            return None
        
        mask_config = self.config_.mask
        # 手动指定路径优先
        if mask_config.path:
            return Path(mask_config.path)
        
        if not mask_config.directory or not self.map_name_:
            if not mask_config.directory:
                logger.warning("未配置掩码文件夹，使用默认空地图")
            if not self.map_name_:
                logger.warning("未提供地图名称，使用默认空地图")
            return None
        
        directory = Path(mask_config.directory)
        candidates = []
        filename_format = mask_config.filename_format or "{map_name}_mask.png"
        try:
            candidates.append(filename_format.format(map_name=self.map_name_))
        except KeyError as exc:
            logger.error(f"掩码命名模板格式错误: {exc}")
        except Exception as exc:
            logger.error(f"渲染掩码命名模板失败: {exc}")
        
        # 兜底命名
        candidates.extend([
            f"{self.map_name_}_mask.png",
            f"{self.map_name_}.png"
        ])
        
        # 去重但保持顺序
        seen = set()
        ordered = []
        for name in candidates:
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        
        for filename in ordered:
            mask_path = directory / filename
            if mask_path.exists():
                logger.info(f"根据地图 {self.map_name_} 匹配掩码: {mask_path}")
                return mask_path
        
        logger.warning(f"未在 {directory} 中找到地图 {self.map_name_} 的掩码，使用默认空地图")
        return None
        
        logger.info("导航主循环结束")
    
    def Stop(self):
        """停止运行"""
        logger.info("正在停止...")
        self.running_ = False
        
        # 停止线程管理器
        if self.thread_manager_:
            self.thread_manager_.stop_all()
        
        # 关闭透明覆盖层
        if self.minimap_service_ and self.minimap_service_.overlay:
            self.minimap_service_.overlay.Close()
        
        logger.info("已停止")

