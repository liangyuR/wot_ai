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

from src.navigation.service.capture_service import CaptureService
from src.vision.minimap_anchor_detector import MinimapAnchorDetector
from src.vision.minimap_detector import MinimapDetector
from src.navigation.service.control_service import ControlService
from src.navigation.core.navigation_executor import NavigationExecutor
from src.navigation.core.mask_loader import load_mask
from src.navigation.service.minimap_service import MinimapService
from src.navigation.service.path_planning_service import PathPlanningService
from src.navigation.nav_runtime.navigation_runtime import NavigationRuntime
from src.navigation.config.models import NavigationConfig


class NavigationInstance:
    """导航实例"""
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(NavigationInstance, cls).__new__(cls)
        return cls.instance

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
        self.nav_runtime_: Optional[NavigationRuntime] = NavigationRuntime()
        
        # 掩码数据
        self.mask_data_ = None
        self.light_initialized_ = False
        self.battle_initialized_ = False
        self.battle_services_ready_ = False
        
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
    
    def set_map_name(self, map_name: Optional[str]) -> None:
        """更新地图名称"""
        self.map_name_ = map_name
    
    def update_mask(self, map_name: Optional[str] = None) -> bool:
        """
        更新地图名称并重新加载掩码
        
        Args:
            map_name: 新的地图名称，如果为None则使用当前地图名称
        
        Returns:
            是否成功更新
        """
        if map_name is not None:
            self.map_name_ = map_name
        
        if not self.map_name_:
            logger.error("未提供地图名称，无法更新掩码")
            return False
        
        if self.running_:
            logger.error("导航系统正在运行，无法更新掩码，请先停止")
            return False
        
        if not self._EnsureBattleServices_():
            logger.error("战斗相关服务未准备，无法更新掩码")
            return False
        
        try:
            logger.info(f"开始更新掩码，地图: {self.map_name_}")
            
            # 1. 检测小地图位置
            logger.info("检测小地图位置...")
            frame = self.capture_service_.Capture()
            if frame is None:
                logger.error("无法捕获屏幕")
                return False
            
            minimap_region = self.minimap_service_.detect_region(frame)
            if minimap_region is None:
                logger.error("无法检测到小地图位置")
                return False
            
            # 2. 重新加载掩码
            minimap_size = (minimap_region['width'], minimap_region['height'])
            mask_path = self._ResolveMaskPath_()
            
            self.mask_data_ = load_mask(mask_path, minimap_size, self.config_.grid.size, self.config_)
            if self.mask_data_ is None:
                logger.error("掩码加载失败")
                return False
            
            # 3. 更新路径规划服务的掩码数据
            if self.path_planning_service_:
                self.path_planning_service_.set_mask_data(
                    self.mask_data_.grid,
                    self.mask_data_.cost_map,
                    self.mask_data_.inflated_obstacle
                )
                logger.info("路径规划服务掩码数据已更新")
            
            # 4. 更新线程管理器的掩码数据（如果存在）
            if self.nag_runtime_:
                self.nag_runtime_.mask_data_ = self.mask_data_
                logger.info("线程管理器掩码数据已更新")
            
            logger.info("掩码更新成功")
            return True
            
        except Exception as e:
            logger.error(f"更新掩码失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _SignalHandler_(self, signum, frame):
        """信号处理器（Ctrl+C）"""
        logger.info("收到退出信号，正在关闭...")
        self.Stop()
    
    def Initialize(self) -> bool:
        """兼容旧接口，仅执行轻量初始化"""
        return self.LightInitialize()
    
    def LightInitialize(self) -> bool:
        """加载与战斗画面无关的基础服务"""
        if self.light_initialized_:
            logger.info("基础导航服务已初始化，跳过轻量初始化")
            return True
        
        try:
            logger.info("开始轻量初始化服务...")
            
            # 1. 初始化YOLO检测器
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
            
            # 2. 初始化控制服务
            logger.info("初始化控制服务...")
            self.control_service_ = ControlService()
            
            # 3. 初始化导航执行器
            logger.info("初始化导航执行器...")
            self.nav_executor_ = NavigationExecutor(
                control_service=self.control_service_,
                move_speed=self.config_.control.move_speed,
                rotation_smooth=self.config_.control.rotation_smooth
            )
            
            # 4. 初始化路径规划服务
            logger.info("初始化路径规划服务...")
            self.path_planning_service_ = PathPlanningService(self.config_)
            
            self.light_initialized_ = True
            logger.info("轻量初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"轻量初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.light_initialized_ = False
            return False
    
    def _EnsureBattleServices_(self) -> bool:
        """确保与战斗画面相关的服务已准备（不启动线程）"""
        if not self.LightInitialize():
            return False
        
        if (
            self.battle_services_ready_
            and self.capture_service_ is not None
            and self.minimap_service_ is not None
        ):
            return True
        
        try:
            logger.info("准备战斗相关服务...")
            
            if self.capture_service_ is None:
                logger.info("初始化屏幕捕获服务...")
                self.capture_service_ = CaptureService(monitor_index=self.config_.monitor_index)
            
            if self.anchor_detector_ is None:
                template_path = str(self.config_.minimap.template_path)
                logger.info(f"初始化小地图锚点检测器: {template_path}")
                self.anchor_detector_ = MinimapAnchorDetector(
                    template_path=template_path,
                    debug=False,
                    multi_scale=False
                )
            
            if self.minimap_service_ is None:
                logger.info("初始化小地图服务...")
                self.minimap_service_ = MinimapService(self.anchor_detector_, self.config_)
            
            self.battle_services_ready_ = True
            logger.info("战斗相关服务准备完成")
            return True
        
        except Exception as e:
            logger.error(f"战斗服务准备失败: {e}")
            import traceback
            traceback.print_exc()
            self.anchor_detector_ = None
            self.minimap_service_ = None
            self.battle_services_ready_ = False
            return False
    
    def BattleInitialize(self) -> bool:
        """战斗阶段初始化：检测小地图、加载掩码、启动线程"""
        if self.battle_initialized_:
            logger.info("战斗阶段服务已就绪，跳过初始化")
            return True
        
        if self.running_:
            logger.warning("导航系统已在运行，无法重复初始化战斗阶段")
            return True
        
        if not self._EnsureBattleServices_():
            return False
        
        try:
            logger.info("开始战斗阶段初始化...")
            
            # # 1. 首次检测小地图位置
            # logger.info("首次检测小地图位置...")
            # frame = self.capture_service_.Capture()
            # if frame is None:
            #     logger.error("无法捕获屏幕")
            #     return False
            
            # minimap_region = self.minimap_service_.detect_region(frame)
            # if minimap_region is None:
            #     logger.error("无法检测到小地图位置")
            #     return False
            
            # 2. 加载掩码
            # minimap_size = (minimap_region['width'], minimap_region['height'])
            # from src.utils.global_path import GetMapMaskPath
            # mask_path = GetMapMaskPath()
            # self.mask_data_ = load_mask(mask_path, minimap_size, self.config_.grid.size, self.config_)
            # if self.mask_data_ is None:
            #     logger.error("掩码加载失败")
            #     return False
            
            # 3. 更新路径规划服务的掩码数据
            # self.path_planning_service_.set_mask_data(
            #     self.mask_data_.grid,
            #     self.mask_data_.cost_map,
            #     self.mask_data_.inflated_obstacle
            # )
            
            # 4. 初始化透明覆盖层
            # if not self.minimap_service_.initialize_overlay(minimap_region):
            #     logger.error("透明覆盖层初始化失败")
            #     return False
            
            # 5. 创建线程管理器并启动
            # queues = {
            #     'detection': self.detection_queue_,
            #     'path': self.path_queue_
            # }
            
            self.nag_runtime_ = NavigationRuntime()
            if not self.nag_runtime_.start():
                logger.error("NavigationRuntime 启动失败")
                self.nag_runtime_ = None
                return False
            
            self.running_ = True
            self.battle_initialized_ = True
            logger.info("战斗阶段初始化完成，线程已启动")
            return True
        
        except Exception as e:
            logger.error(f"战斗阶段初始化失败: {e}")
            import traceback
            traceback.print_exc()
            if self.nag_runtime_:
                self.nag_runtime_.stop_all()
                self.nag_runtime_ = None
            self.running_ = False
            self.battle_initialized_ = False
            return False
    
    def Run(self):
        """主循环：启动多线程架构"""
        logger.info("启动导航主循环（多线程模式）...")
        
        if not self.BattleInitialize():
            logger.error("战斗阶段初始化失败，无法启动导航主循环")
            return
        
        logger.info("所有线程已启动，等待线程运行...")
        
        # 主线程等待所有线程完成
        try:
            while self.running_ and self.nag_runtime_ and self.nag_runtime_.is_running():
                time.sleep(5)
                # 检查线程是否还在运行
                if (
                    self.nag_runtime_.detection_thread_
                    and not self.nag_runtime_.detection_thread_.is_alive()
                ):
                    logger.warning("检测线程已退出")
                if (
                    self.nag_runtime_.control_thread_
                    and not self.nag_runtime_.control_thread_.is_alive()
                ):
                    logger.warning("控制线程已退出")
                if (
                    self.nag_runtime_.ui_thread_
                    and not self.nag_runtime_.ui_thread_.is_alive()
                ):
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
    
    def Stop(self):
        """停止运行"""
        logger.info("正在停止...")
        self.running_ = False
        
        # 停止线程管理器
        if self.nag_runtime_:
            self.nag_runtime_.stop_all()
            self.nag_runtime_ = None
        
        # 关闭透明覆盖层
        if self.minimap_service_ and self.minimap_service_.overlay:
            self.minimap_service_.overlay.Close()
        
        self.battle_initialized_ = False
        logger.info("已停止")

