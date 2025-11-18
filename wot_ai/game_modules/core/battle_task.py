#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
战斗任务模块

实现单局战斗的完整流程。
"""

import time
from typing import Optional
from loguru import logger
import pyautogui

from .state_machine import StateMachine, GameState
from .actions import wait, wait_state
from .ai_controller import AIController
from .tank_selector import TankSelector, TankTemplate
from wot_ai.game_modules.ui_control.actions import UIActions
from wot_ai.game_modules.navigation.config.models import NavigationConfig
from wot_ai.game_modules.vision.detection.map_name_detector import MapNameDetector
from wot_ai.game_modules.core.actions import screenshot_with_key_hold

from pynput import keyboard

class BattleTask:
    """战斗任务"""
    
    def __init__(
        self,
        tank_selector: TankSelector,
        state_machine: StateMachine,
        map_detector: MapNameDetector,
        ai_config: NavigationConfig,
        ui_actions: UIActions,
        selection_retry_interval: float = 10.0,
        selection_timeout: float = 120.0
    ):
        """
        初始化战斗任务
        
        Args:
            tank_selector: 坦克选择器
            state_machine: 状态机实例
            map_detector: 地图名称识别器
            ai_config: NavigationConfig配置对象
            ui_actions: UI 控制对象
            selection_retry_interval: 选择失败后的重试间隔
            selection_timeout: 选择超时时间
        """
        self.tank_selector_ = tank_selector
        self.state_machine_ = state_machine
        self.map_detector_ = map_detector
        self.ai_config_ = ai_config
        self.ai_controller_ = AIController()
        self.ui_actions_ = ui_actions
        self.selection_retry_interval_ = selection_retry_interval
        self.selection_timeout_ = selection_timeout
        self.selected_tank_: Optional[TankTemplate] = None

    def run(self) -> bool:
        """
        执行战斗任务
        
        Returns:
            是否成功完成
        """
        try:
            logger.info("开始战斗任务")
            
            # 1. 选择车辆
            if not self.select_tank():
                logger.error("选择车辆失败")
                return False
            
            if self.selected_tank_ is not None:
                logger.info(f"当前车辆: {self.selected_tank_.name}")
            
            # 2. 进入战斗
            if not self.enter_battle():
                logger.error("进入战斗失败")
                return False
            
            # 3. 等待加载并识别地图
            map_name = self.wait_loading()
            if not map_name:
                logger.warning("等待加载或识别地图失败，使用默认地图继续")
                map_name = "default"
            
            # 4. 战斗循环（启动AI，监控状态）
            if not self.in_battle_loop(map_name):
                logger.error("战斗循环失败")
                return False
            
            # 5. 战斗结束后，返回车库
            if not self.enter_grage():
                logger.error("返回车库失败")
                return False
            
            logger.info("战斗任务完成")
            return True
            
        except Exception as e:
            logger.error(f"战斗任务执行异常: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 确保停止AI控制器
            self.ai_controller_.stop()
    
    def select_tank(self) -> bool:
        """
        选择车辆
        
        Returns:
            是否成功选择
        """
        logger.info("选择车辆...")
        
        deadline = time.time() + self.selection_timeout_
        
        while time.time() < deadline:
            candidates = self.tank_selector_.pick()
            if not candidates:
                logger.error("车辆模板列表为空，无法选择")
                return False
            
            for candidate in candidates:
                if self._try_select_candidate(candidate):
                    self.selected_tank_ = candidate
                    logger.info(f"车辆选择成功: {candidate.name}")
                    return True
            
            logger.info(f"所有车辆暂不可用，等待 {self.selection_retry_interval_} 秒后重试")
            wait(self.selection_retry_interval_)
        
        logger.error("选择车辆超时")
        return False
    
    def _try_select_candidate(self, candidate: TankTemplate) -> bool:
        """尝试点击单个候选车辆"""
        success = self.ui_actions_.SelectVehicle(
            template_name=candidate.name,
            template_dir=str(candidate.directory),
            confidence=candidate.confidence
        )
        if success:
            wait(0.5)
        else:
            logger.debug(f"车辆不可用或未找到: {candidate.name}")
        return success
    
    def enter_battle(self) -> bool:
        """
        点击加入战斗
        
        Returns:
            是否成功
        """
        logger.info("点击加入战斗...")
        
        success = self.ui_actions_.ClickTemplate(
            "join_battle.png",
            timeout=5.0,
            confidence=0.85,
            max_retries=3
        )
        if not success:
            logger.error("未找到加入战斗按钮")
            return False
        
        wait(1.0)  # 等待进入加载界面
        logger.info("已点击加入战斗")
        return True
    
    def wait_loading(self) -> Optional[str]:
        """
        等待加载完成并识别地图名称
        
        Returns:
            地图名称，如果失败则返回None
        """
        logger.info("等待战斗加载...")
        
        # 先等待加载状态，若失败则记录告警继续
        if not wait_state(self.state_machine_, GameState.IN_BATTLE, timeout=180.0):
            logger.error("等待进入战斗超时")
            return None
        
        logger.info("已进入战斗，开始识别地图名称... 10秒后开始识别")
        map_name = self.map_detector_.detect()
        if map_name:
            logger.info(f"识别到地图: {map_name}")
        else:
            logger.warning("地图名称识别失败，使用默认地图配置")
            map_name = "default"
        
        return map_name

    def in_battle_loop(self, map_name: str) -> bool:
        """
        战斗循环：启动AI，监控状态
        
        Args:
            map_name: 地图名称
        
        Returns:
            是否成功完成战斗
        """
        logger.info("启动AI控制器...")
        
        # 启动AI控制器
        if not self.ai_controller_.start(self.ai_config_, map_name):
            logger.error("AI控制器启动失败")
            return False
        
        # 监控状态，直到战斗结束
        logger.info("监控战斗状态...")
        while self.ai_controller_.is_running():
            self.state_machine_.update()
            current_state = self.state_machine_.current_state()

            # 检查是否被击毁或战斗结束
            if current_state == GameState.IN_END:
                logger.info("战斗结束")
                self.ai_controller_.stop()
                break

            wait(5)  # 检查间隔
        
        return True
    
    def wait_battle_result(self) -> bool:
        """
        等待结算界面出现
        
        Returns:
            是否成功
        """
        logger.info("等待结算界面...")
        return wait_state(self.state_machine_, GameState.BATTLE_RESULT, timeout=30.0)
    
    def enter_grage(self) -> bool:
        """
        退出结算界面，返回车库
        
        Returns:
            是否成功
        """
        logger.info("退出结算界面...")

        if GameState.IN_RESULT_PAGE == self.state_machine_.current_state():
            screenshot_with_key_hold(keyboard.Key.esc, hold_duration=0.5, warmup=0.0)

        # 按下 esc 按键
        screenshot_with_key_hold(keyboard.Key.esc, hold_duration=0.5, warmup=0.0)

        # success = self.ui_actions_.WaitAppear(
        #     "return_garage.png",
        #     timeout=5.0,
        #     confidence=0.85,
        #     max_retries=3
        # )
        # if not success:
        #     logger.error("未找到返回车库按钮")
        #     return False

        # 点击"返回车库"按钮
        success = self.ui_actions_.ClickTemplate(
            "return_garage.png",
            timeout=5.0,
            confidence=0.85,
            max_retries=3
        )
        if not success:
            logger.error("未找到继续按钮")
            return False

        # # 等待出现离开确认按钮
        # success = self.ui_actions_.WaitAppear(
        #     "leave_confirmation.png",
        #     timeout=5.0,
        #     confidence=0.85,
        #     max_retries=3
        # )
        # if not success:
        #     logger.error("未找到离开确认按钮")
        #     return False
        
        # 点击离开确认按钮
        success = self.ui_actions_.ClickTemplate(
            "leave_confirmation.png",
            timeout=5.0,
            confidence=0.85,
            max_retries=3
        )
        if not success:
            logger.error("未找到离开确认按钮")
            return False
        
        logger.info("已返回车库")
        return True