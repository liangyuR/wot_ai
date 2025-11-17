#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏状态机模块

基于模板匹配识别游戏状态，包含稳定性机制。
"""

import time
from enum import Enum
from typing import Optional
import numpy as np
from loguru import logger

from wot_ai.game_modules.ui_control.matcher_pyautogui import match_template


class GameState(Enum):
    """游戏状态枚举"""
    GARAGE = "garage"
    BATTLE_LOADING = "battle_loading"
    IN_BATTLE = "in_battle"
    DESTROYED = "destroyed"
    BATTLE_RESULT = "battle_result"
    RETURN_TO_GARAGE = "return_to_garage"
    UNKNOWN = "unknown"


class StateMachine:
    """游戏状态机"""
    
    def __init__(self, confirmation_frames: int = 3):
        """
        初始化状态机
        
        Args:
            confirmation_frames: 状态确认所需的连续帧数，默认3帧
        """
        self.confirmation_frames_ = confirmation_frames
        self.current_state_ = GameState.UNKNOWN
        self.state_history_ = []  # 状态历史记录
        self.last_update_time_ = 0.0
        
        # 状态模板映射
        self.state_templates_ = {
            GameState.GARAGE: "garage_join_battle.png",
            GameState.BATTLE_LOADING: "battle_loading.png",
            GameState.IN_BATTLE: "minimap_anchor.png",
            GameState.DESTROYED: "destroyed_icon.png",
            GameState.BATTLE_RESULT: "battle_result_continue.png",
            GameState.RETURN_TO_GARAGE: "return_garage_loading.png",
        }
    
    def update(self, frame: Optional[np.ndarray] = None) -> GameState:
        """
        更新状态机
        
        Args:
            frame: 可选的屏幕截图（BGR格式），如果为None则自动截图
        
        Returns:
            当前确认的游戏状态
        """
        # 检测所有可能的状态
        detected_states = []
        
        for state, template_name in self.state_templates_.items():
            center = match_template(template_name, confidence=0.75)
            if center is not None:
                detected_states.append(state)
        
        # 确定当前检测到的状态（优先级：战斗中 > 其他状态）
        detected_state = GameState.UNKNOWN
        if GameState.IN_BATTLE in detected_states:
            detected_state = GameState.IN_BATTLE
        elif GameState.DESTROYED in detected_states:
            detected_state = GameState.DESTROYED
        elif GameState.BATTLE_RESULT in detected_states:
            detected_state = GameState.BATTLE_RESULT
        elif GameState.BATTLE_LOADING in detected_states:
            detected_state = GameState.BATTLE_LOADING
        elif GameState.RETURN_TO_GARAGE in detected_states:
            detected_state = GameState.RETURN_TO_GARAGE
        elif GameState.GARAGE in detected_states:
            detected_state = GameState.GARAGE
        
        # 更新状态历史
        self.state_history_.append(detected_state)
        
        # 保持历史记录长度（只保留最近N帧）
        max_history = self.confirmation_frames_ * 2
        if len(self.state_history_) > max_history:
            self.state_history_ = self.state_history_[-max_history:]
        
        # 检查是否有足够连续的状态确认
        if len(self.state_history_) >= self.confirmation_frames_:
            recent_states = self.state_history_[-self.confirmation_frames_:]
            if all(s == detected_state for s in recent_states):
                # 状态确认，更新当前状态
                if self.current_state_ != detected_state:
                    logger.info(f"状态切换: {self.current_state_.value} -> {detected_state.value}")
                    self.current_state_ = detected_state
            else:
                # 状态不稳定，返回Unknown
                if self.current_state_ != GameState.UNKNOWN:
                    logger.debug(f"状态不稳定，返回Unknown（检测到: {detected_state.value}）")
                self.current_state_ = GameState.UNKNOWN
        
        self.last_update_time_ = time.time()
        return self.current_state_
    
    def current_state(self) -> GameState:
        """
        获取当前确认的状态
        
        Returns:
            当前游戏状态
        """
        return self.current_state_
    
    def wait_state(
        self,
        target_state: GameState,
        timeout: float = 10.0,
        check_interval: float = 3
    ) -> bool:
        """
        等待状态切换到目标状态
        
        Args:
            target_state: 目标状态
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）
        
        Returns:
            是否在超时前达到目标状态
        """
        start_time = time.time()
        logger.info(f"等待状态切换到: {target_state.value} (timeout={timeout}s)")
        
        while time.time() - start_time < timeout:
            self.update()
            if self.current_state_ == target_state:
                logger.info(f"已到达目标状态: {target_state.value}")
                return True
            time.sleep(check_interval)
        
        logger.warning(f"等待状态超时: {target_state.value} (timeout={timeout}s)")
        return False

