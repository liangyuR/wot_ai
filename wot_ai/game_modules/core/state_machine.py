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
import cv2
from loguru import logger
from pathlib import Path

from wot_ai.config import get_program_dir
from wot_ai.game_modules.core.actions import screenshot
from wot_ai.game_modules.core.global_context import GlobalContext


class GameState(Enum):
    """游戏状态枚举"""
    IN_GARAGE = "in_garage"
    IN_LOADING = "in_loading"
    IN_BATTLE = "in_battle"
    IN_END = "in_end"
    IN_RESULT_PAGE = "in_result_page"
    UNKNOWN = "unknown"


class StateMachine:
    """游戏状态机"""
    
    def __init__(
        self,
        confirmation_frames: int = 3,
        global_context: Optional[GlobalContext] = None
    ):
        """
        初始化状态机
        
        Args:
            confirmation_frames: 状态确认所需的连续帧数，默认3帧
        """
        self.confirmation_frames_ = confirmation_frames
        self.current_state_ = GameState.UNKNOWN
        self.state_history_ = []  # 状态历史记录
        self.last_update_time_ = 0.0
        self.global_context_ = global_context or GlobalContext()
        
        # 状态模板映射
        self.state_templates_ = {
            GameState.IN_GARAGE: "in_garage.png",
            GameState.IN_LOADING: "in_loading.png",
            GameState.IN_END: "pingjia.png",
            GameState.IN_BATTLE: "in_battle.png",
            # TODO(@liangyu) 尝试判定评价系统模板，貌似每一局结束后，都会出现评价模板（无论胜利/失败/平局），那么当评价模板出现时，则可以判定游戏结束
            GameState.IN_RESULT_PAGE: "space_jump.png", # 在胜利/失败结算页面（偶尔可能不会直接回到车库）
        }
        
        self.template_resolution_ = self.global_context_.template_tier
        logger.info(
            f"使用模板目录: {self.template_resolution_} "
            f"(分辨率: {self.global_context_.resolution[0]}x{self.global_context_.resolution[1]})"
        )
    
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

    def detect_state(self, frame: Optional[np.ndarray] = None) -> GameState:
        """
        检测游戏状态
        
        Args:
            frame: 可选的屏幕截图（BGR格式），如果为None则自动截图
        
        Returns:
            当前检测到的游戏状态
        """
        if frame is None:
            frame = screenshot()
        if frame is None:
            logger.warning("无法获取截图，跳过状态检测")
            return GameState.UNKNOWN

        # 统一处理模板列表（字符串和列表都支持）
        for state, templates in self.state_templates_.items():
            # 将单个字符串转换为列表以便统一处理
            template_list = templates if isinstance(templates, list) else [templates]
            
            for template_name in template_list:
                # 构建模板路径
                template_path = get_program_dir() / "resource" / "template" / self.template_resolution_ / template_name
                
                if not template_path.exists():
                    logger.debug(f"模板文件不存在: {template_path}")
                    continue
                
                # 加载模板图像
                template = cv2.imread(str(template_path))
                if template is None:
                    logger.warning(f"无法加载模板图像: {template_path}")
                    continue
                
                # 使用 OpenCV 进行模板匹配
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                # 设置匹配阈值
                threshold = 0.8
                if max_val >= threshold:
                    logger.debug(f"检测到状态: {state.value} (模板: {template_name}, 匹配度: {max_val:.2f})")
                    return state
        
        return GameState.UNKNOWN


if __name__ == "__main__":
    frame = screenshot()
    context = GlobalContext()
    sm = StateMachine(confirmation_frames=2, global_context=context)
    state = sm.detect_state(frame)
    print(f"当前检测到的状态: {state.value}")