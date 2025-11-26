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
from src.utils.screen_action import ScreenAction
from src.utils.template_matcher import TemplateMatcher

class GameState(Enum):
    """游戏状态枚举"""
    IN_GARAGE = "in_garage"
    IN_BATTLE = "in_battle"
    IN_END = "in_end"
    IN_RESULT_PAGE = "in_result_page"
    UNKNOWN = "unknown"


class StateMachine:
    """游戏状态机"""
    
    _instance: Optional['StateMachine'] = None
    
    def __new__(cls) -> 'StateMachine':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        confirmation_frames: int = 1,
    ):
        """
        初始化状态机
        
        Args:
            confirmation_frames: 状态确认所需的连续帧数，默认3帧
        """
        # 如果已经初始化过，跳过
        if hasattr(self, '_initialized'):
            return
        
        self.confirmation_frames_ = confirmation_frames
        self.current_state_ = GameState.UNKNOWN
        self.state_history_ = []
        self.last_update_time_ = 0.0
        self.template_matcher_ = TemplateMatcher()
        self.screen_action_ = ScreenAction()
        
        # 状态模板映射
        self.state_templates_ = {
            GameState.IN_GARAGE: "in_garage.png",
            GameState.IN_END: "pingjia.png", # 被击毁时会出现 “评价窗口”，当出现评价窗口时，认为此时已经被击毁
            GameState.IN_BATTLE: "in_battle.png",
            GameState.IN_RESULT_PAGE: "result_page.png", # 在胜利/失败后的结算页面
            # TODO(@ly) 勋章获取页面没有hack
        }
        self._initialized = True
    
    def update(self, frame: Optional[np.ndarray] = None) -> GameState:
        """
        更新状态机
        
        Args:
            frame: 可选的屏幕截图（BGR格式），如果为None则自动截图
        
        Returns:
            当前确认的游戏状态
        """
        # 检测所有可能的状态
        detected_state = self.detect_state(frame)
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
        check_interval: float = 1
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

    def detect_state(self, frame: Optional[np.ndarray] = None) -> GameState:
        """
        检测游戏状态
        
        Args:
            frame: 可选的屏幕截图（BGR格式），如果为None则自动截图
        
        Returns:
            当前检测到的游戏状态
        """
        if frame is None:
            frame = ScreenAction.screenshot()

        if frame is None:
            logger.warning("无法获取截图，跳过状态检测")
            return GameState.UNKNOWN
        
        for state, templates in self.state_templates_.items():
            for template_name in (templates if isinstance(templates, list) else [templates]):
                if self.template_matcher_.match_template(template_name, confidence=0.80):
                    logger.debug(f"检测到状态: {state.value} (模板: {template_name})")
                    return state

        return GameState.UNKNOWN


if __name__ == "__main__":
    state_machine = StateMachine()
    state_machine.update()
    print(state_machine.current_state())
    state_machine.wait_state(GameState.IN_GARAGE)
    print(state_machine.current_state())
    state_machine.wait_state(GameState.IN_BATTLE)
    print(state_machine.current_state())
    state_machine.wait_state(GameState.IN_END)
    print(state_machine.current_state())
    state_machine.wait_state(GameState.IN_RESULT_PAGE)
    print(state_machine.current_state())