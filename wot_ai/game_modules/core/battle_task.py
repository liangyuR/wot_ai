#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
战斗任务模块

实现单局战斗的完整流程。
"""

import time
from typing import Optional
from loguru import logger
import numpy as np
import pyautogui

from .state_machine import StateMachine, GameState
from .actions import wait, wait_state, screenshot
from .map_name_detector import MapNameDetector
from .ai_controller import AIController
from .tank_selector import TankSelector, TankTemplate
from wot_ai.game_modules.ui_control.actions import UIActions
from wot_ai.game_modules.navigation.config.models import NavigationConfig
from wot_ai.data_collection.game_state_detector import GameStateDetector


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
        self.game_state_detector_ = GameStateDetector()
    
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
                logger.error("等待加载或识别地图失败")
                return False
            
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
        if wait_state(self.state_machine_, GameState.IN_LOADING, timeout=60.0):
            logger.info("已检测到加载界面，继续等待进入战斗")
        else:
            logger.warning("未能确认加载界面，继续等待进入战斗")
        
        if not wait_state(self.state_machine_, GameState.IN_BATTLE, timeout=180.0):
            logger.error("等待进入战斗超时")
            return None
        
        logger.info("已进入战斗，开始识别地图名称... 10秒后开始识别")
        wait(10.0)
        map_name = self.map_detector_.detect()
        if map_name:
            logger.info(f"识别到地图: {map_name}")
        else:
            logger.warning("地图名称识别失败")
            return None
        
        return map_name

    def _press_key(self, key: str, hold: float = 0.1) -> bool:
        """
        按下并释放指定按键
        """
        if pyautogui is None:
            logger.error("pyautogui 未安装，无法发送按键")
            return False
        
        try:
            pyautogui.keyDown(key)
            if hold > 0:
                wait(hold)
            pyautogui.keyUp(key)
            logger.info(f"已按下按键: {key}")
            return True
        except Exception as exc:
            logger.error(f"发送按键 {key} 失败: {exc}")
            return False

    def _handle_manual_exit_to_garage(self, reason: str) -> None:
        """
        通过按ESC和点击return按钮返回车库
        """
        logger.info(f"检测到战斗结束状态({reason})，执行手动退出流程")
        
        if not self._press_key('esc', hold=1):
            logger.warning("无法按下 ESC，尝试继续退出流程")
        else:
            wait(1.0)
        
        clicked = self.ui_actions_.ClickTemplate(
            "return.png",
            timeout=15.0,
            confidence=0.85,
            max_retries=5
        )
        if not clicked:
            logger.warning("未找到 return 按钮，可能已返回车库或UI变化")
        
        if not wait_state(self.state_machine_, GameState.GARAGE, timeout=60.0):
            logger.warning("手动退出流程等待返回车库超时")
        else:
            logger.info("已通过手动流程回到车库")

    def _attempt_detect_battle_end(self) -> bool:
        """
        OCR 检测胜利/失败/被击毁等状态
        """
        frame = screenshot()
        if frame is None:
            logger.debug("无法获取截图，跳过战斗结果检测")
            return False
        
        detected_state = self.game_state_detector_.DetectState(frame)
        if detected_state in {'战斗胜利', '战斗失败', '坦克被该玩家击毁'}:
            logger.info(f"OCR 检测到战斗状态: {detected_state}")
            self.ai_controller_.stop()
            self._handle_manual_exit_to_garage(detected_state)
            return True
        
        return False
    
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
        
        # 点击"继续"按钮
        success = self.ui_actions_.ClickTemplate(
            "battle_result_continue.png",
            timeout=5.0,
            confidence=0.85,
            max_retries=3
        )
        if not success:
            logger.error("未找到继续按钮")
            return False
        
        wait(2.0)  # 等待返回车库
        
        # 等待返回车库
        if not wait_state(self.state_machine_, GameState.GARAGE, timeout=30.0):
            logger.error("等待返回车库超时")
            return False
        
        logger.info("已返回车库")
        return True