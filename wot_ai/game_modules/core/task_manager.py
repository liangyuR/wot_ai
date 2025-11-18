#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务管理器模块

持续执行多局战斗循环，处理UI配置的停止条件。
"""

import time
import os
import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
from loguru import logger

from .state_machine import StateMachine
from .global_context import GlobalContext
from wot_ai.game_modules.vision.detection.map_name_detector import MapNameDetector
from .tank_selector import TankSelector
from .battle_task import BattleTask
from wot_ai.game_modules.ui_control.actions import UIActions
from wot_ai.game_modules.navigation.config.models import NavigationConfig


class TaskManager:
    """任务管理器"""
    
    def __init__(
        self,
        vehicle_screenshot_dir: Path,
        vehicle_priority: List[str],
        ai_config: NavigationConfig,
        run_hours: int = 4,
        auto_stop: bool = False,
        auto_shutdown: bool = False,
        global_context: Optional[GlobalContext] = None
    ):
        """
        初始化任务管理器
        
        Args:
            vehicle_screenshot_dir: 车辆截图目录
            vehicle_priority: 车辆优先级列表
            ai_config: NavigationConfig配置对象
            run_hours: 运行时长限制（小时）
            auto_stop: 达到时长后自动停止
            auto_shutdown: 达到时长后自动关机（需要管理员权限）
            global_context: 全局上下文（分辨率、模板信息）
        """
        self.vehicle_screenshot_dir_ = vehicle_screenshot_dir
        self.vehicle_priority_ = vehicle_priority
        self.ai_config_ = ai_config
        self.run_hours_ = run_hours
        self.auto_stop_ = auto_stop
        self.auto_shutdown_ = auto_shutdown
        self.global_context_ = global_context or GlobalContext()
        
        self.running_ = False
        self.start_time_ = None
        self.state_machine_ = StateMachine(global_context=self.global_context_)
        self.map_detector_ = MapNameDetector()
        self.tank_selector_ = TankSelector(vehicle_screenshot_dir, vehicle_priority)
        self.ui_actions_ = UIActions()
    
    def run_forever(self) -> None:
        """持续执行战斗循环"""
        self.running_ = True
        self.start_time_ = datetime.now()
        
        logger.info(f"任务管理器启动，运行时长限制: {self.run_hours_} 小时")
        
        try:
            while self.running_:
                # 检查停止条件
                if self._should_stop():
                    logger.info("达到停止条件，退出循环")
                    break
                
                # 创建战斗任务
                task = BattleTask(
                    self.tank_selector_,
                    self.state_machine_,
                    self.map_detector_,
                    self.ai_config_,
                    self.ui_actions_
                )
                
                # 执行战斗任务
                success = task.run()
                if not success:
                    logger.warning("战斗任务执行失败，继续下一局")
                
                # 短暂休息
                time.sleep(2.0)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止任务管理器")
        except Exception as e:
            logger.error(f"任务管理器运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running_ = False
            logger.info("任务管理器已停止")
            
            # 如果配置了自动关机
            if self.auto_shutdown_ and self._should_stop():
                self._shutdown()
    
    def stop(self) -> None:
        """停止任务管理器"""
        logger.info("正在停止任务管理器...")
        self.running_ = False
    
    def _should_stop(self) -> bool:
        """
        检查是否应该停止
        
        Returns:
            是否应该停止
        """
        if not self.auto_stop_ and not self.auto_shutdown_:
            return False
        
        if self.start_time_ is None:
            return False
        
        elapsed = datetime.now() - self.start_time_
        elapsed_hours = elapsed.total_seconds() / 3600.0
        
        return elapsed_hours >= self.run_hours_
    
    def _shutdown(self) -> None:
        """执行关机操作（需要管理员权限）"""
        logger.info("执行自动关机...")
        
        try:
            subprocess.run(['shutdown', '/s', '/t', '10'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"关机命令执行失败: {e}")
        except Exception as e:
            logger.error(f"关机操作异常: {e}")

