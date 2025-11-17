#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
战斗任务模块

实现单局战斗的完整流程。
"""

import time
from pathlib import Path
from typing import Optional
from loguru import logger

import pyautogui

from .state_machine import StateMachine, GameState
from .actions import wait, wait_state, screenshot
from .map_name_detector import MapNameDetector
from .ai_controller import AIController
from wot_ai.game_modules.ui_control.matcher_pyautogui import match_template
from wot_ai.game_modules.navigation.config.models import NavigationConfig
from wot_ai.config import get_program_dir


class BattleTask:
    """战斗任务"""
    
    def __init__(
        self,
        tank_image_path: Path,
        state_machine: StateMachine,
        map_detector: MapNameDetector,
        ai_config: NavigationConfig
    ):
        """
        初始化战斗任务
        
        Args:
            tank_image_path: 车辆截图路径
            state_machine: 状态机实例
            map_detector: 地图名称识别器
            ai_config: NavigationConfig配置对象
        """
        self.tank_image_path_ = tank_image_path
        self.state_machine_ = state_machine
        self.map_detector_ = map_detector
        self.ai_config_ = ai_config
        self.ai_controller_ = AIController()
    
    def run(self) -> bool:
        """
        执行战斗任务
        
        Returns:
            是否成功完成
        """
        try:
            logger.info(f"开始战斗任务，车辆: {self.tank_image_path_.name}")
            
            # 1. 选择车辆
            if not self.select_tank():
                logger.error("选择车辆失败")
                return False
            
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
            
            # 5. 等待结算界面
            if not self.wait_battle_result():
                logger.error("等待结算界面失败")
                return False
            
            # 6. 退出结算界面
            if not self.exit_result_screen():
                logger.error("退出结算界面失败")
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
        
        # 使用模板匹配找到车辆位置
        center = match_template(self.tank_image_path_.name, confidence=0.75, template_dir=str(get_program_dir() / "vehicle_screenshots"))
        if center is None:
            logger.error(f"未找到车辆: {self.tank_image_path_.name}")
            return False
        
        # 直接使用pyautogui点击
        pyautogui.click(center[0], center[1])
        wait(0.5)  # 等待选择生效
        
        logger.info("车辆选择成功")
        return True
    
    def enter_battle(self) -> bool:
        """
        点击加入战斗
        
        Returns:
            是否成功
        """
        logger.info("点击加入战斗...")
        
        # 使用模板匹配找到"加入战斗"按钮
        center = match_template("garage_join_battle.png", confidence=0.85)
        if center is None:
            logger.error("未找到加入战斗按钮")
            return False
        
        # 直接使用pyautogui点击
        pyautogui.click(center[0], center[1])
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
        
        # 等待状态切换到 InBattle
        if not wait_state(self.state_machine_, GameState.IN_BATTLE, timeout=40.0):
            logger.error("等待进入战斗超时")
            return None
        
        logger.info("已进入战斗，开始识别地图名称...")
        
        # 截图并识别地图名称
        frame = screenshot()
        if frame is None:
            logger.error("截图失败")
            return None
        
        map_name = self.map_detector_.detect(frame)
        if map_name:
            logger.info(f"识别到地图: {map_name}")
        else:
            logger.warning("地图名称识别失败，使用默认配置")
            map_name = "default"  # 使用默认地图名称
        
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
            if current_state == GameState.DESTROYED:
                logger.info("车辆被击毁")
                self.ai_controller_.stop()
                break
            
            if current_state == GameState.BATTLE_RESULT:
                logger.info("战斗结束")
                self.ai_controller_.stop()
                break
            
            wait(0.5)  # 检查间隔
        
        return True
    
    def wait_battle_result(self) -> bool:
        """
        等待结算界面出现
        
        Returns:
            是否成功
        """
        logger.info("等待结算界面...")
        return wait_state(self.state_machine_, GameState.BATTLE_RESULT, timeout=30.0)
    
    def exit_result_screen(self) -> bool:
        """
        退出结算界面，返回车库
        
        Returns:
            是否成功
        """
        logger.info("退出结算界面...")
        
        # 点击"继续"按钮
        center = match_template("battle_result_continue.png", confidence=0.85)
        if center is None:
            logger.error("未找到继续按钮")
            return False
        
        pyautogui.click(center[0], center[1])
        wait(2.0)  # 等待返回车库
        
        # 等待返回车库
        if not wait_state(self.state_machine_, GameState.GARAGE, timeout=30.0):
            logger.error("等待返回车库超时")
            return False
        
        logger.info("已返回车库")
        return True

