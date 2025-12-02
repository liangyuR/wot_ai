#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
战斗任务模块

基于 hub state 的事件驱动模式，实现自动战斗循环。
"""

import time
import threading
from typing import Optional
from loguru import logger

from src.core.state_machine import StateMachine, GameState
from src.core.tank_selector import TankSelector, TankTemplate
from src.vision.map_name_detector import MapNameDetector
from src.utils.template_matcher import TemplateMatcher
from src.utils.key_controller import KeyController
from src.navigation.nav_runtime.navigation_runtime import NavigationRuntime
from pynput.keyboard import Key
from src.utils.restart_game import GameRestarter
from src.utils.global_path import GetGlobalConfig


class BattleTask:
    """战斗任务（事件驱动模式）"""
    
    def __init__(
        self,
        selection_retry_interval: float = 10.0,
        selection_timeout: float = 120.0,
        state_check_interval: float = 3
    ):
        """
        初始化战斗任务
        
        Args:
            config: 配置对象
            selection_retry_interval: 选择失败后的重试间隔
            selection_timeout: 选择超时时间
            state_check_interval: 状态检测间隔（秒）
        """
        self.tank_selector_ = TankSelector()
        self.state_machine_ = StateMachine()
        self.template_matcher_ = TemplateMatcher()
        self.map_name_detector_ = MapNameDetector()
        self.key_controller_ = KeyController()
        self.navigation_runtime_ = None
        self.navigation_thread_ = None
        
        config = GetGlobalConfig()
        self.game_restarter_ = GameRestarter(
            process_name=config.game.process_name,
            game_exe_path=config.game.exe_path,
            wait_seconds=config.game.restart_wait_seconds
        )
        self.stuck_timeout_ = config.game.stuck_timeout_seconds

        self.selection_retry_interval_ = selection_retry_interval
        self.selection_timeout_ = selection_timeout
        self.state_check_interval_ = state_check_interval
        
        # 事件驱动相关
        self.running_ = False
        self.event_thread_: Optional[threading.Thread] = None
        
        # 状态处理标志，防止重复处理
        self.garage_handled_ = False
        self.battle_handled_ = False
        self.end_handled_ = False
        self.result_page_handled_ = False
        
        # 选中的车辆
        self.selected_tank_: Optional[TankTemplate] = None
        
        # 状态监测
        self.last_state_ = GameState.UNKNOWN
        self.last_state_change_time_ = time.time()
    
    def start(self) -> bool:
        """
        启动事件驱动循环
        
        Returns:
            是否成功启动
        """
        if self.running_:
            logger.warning("事件驱动循环已在运行")
            return False
        
        logger.info("启动事件驱动循环...")
        self.running_ = True
        
        # 重置状态标志
        self.garage_handled_ = False
        self.battle_handled_ = False
        self.end_handled_ = False
        self.result_page_handled_ = False
        
        # 启动事件循环线程
        self.event_thread_ = threading.Thread(target=self._event_loop, daemon=True)
        self.event_thread_.start()
        
        logger.info("事件驱动循环已启动")
        return True
    
    def stop(self) -> None:
        """
        停止事件驱动循环和导航AI
        """
        if not self.running_:
            return
        
        logger.info("正在停止事件驱动循环...")
        self.running_ = False
        
        # 停止导航AI
        if self.navigation_thread_ and self.navigation_thread_.is_alive():
            self.navigation_thread_.join(timeout=3.0)
        
        # 等待事件循环线程结束
        if self.event_thread_ and self.event_thread_.is_alive():
            self.event_thread_.join(timeout=3.0)
        
        logger.info("事件驱动循环已停止")
    
    def _event_loop(self) -> None:
        """
        事件驱动主循环
        """
        logger.info("事件循环开始运行")
        self.last_state_change_time_ = time.time()
        self.last_state_ = GameState.UNKNOWN
        
        while self.running_:
            try:
                # 更新状态机
                self.state_machine_.update()
                current_state = self.state_machine_.current_state()
                
                # 状态变化检测
                if current_state != self.last_state_:
                    self.last_state_ = current_state
                    self.last_state_change_time_ = time.time()
                
                # 卡死检测
                elapsed_since_change = time.time() - self.last_state_change_time_
                if elapsed_since_change > self.stuck_timeout_:
                    logger.error(f"游戏状态已 {elapsed_since_change:.1f} 秒未变化，判定为卡死，执行重启...")
                    self.game_restarter_.RestartGame()
                    # 重启后重置状态计时
                    self.last_state_change_time_ = time.time()
                    self.last_state_ = GameState.UNKNOWN
                    # 等待游戏启动
                    time.sleep(30)
                    continue

                # 根据当前状态执行相应操作
                if current_state == GameState.IN_GARAGE:
                    self._handle_garage_state()
                elif current_state == GameState.IN_BATTLE:
                    self._handle_battle_state()
                elif current_state == GameState.IN_END:
                    self._handle_end_state()
                elif current_state == GameState.UNKNOWN:
                    pass
                
                # 等待一段时间后再次检查
                time.sleep(self.state_check_interval_)
                
            except Exception as e:
                logger.error(f"事件循环异常: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(self.state_check_interval_)
        
        logger.info("事件循环结束")
    
    def _handle_garage_state(self) -> None:
        """
        处理车库状态：选择车辆并加入战斗
        """
        if self.garage_handled_:
            return
        logger.info("检测到车库状态，开始选择车辆...")

        self.end_handled_ = False
        self.battle_handled_ = False

        # 选择车辆
        if not self.select_tank():
            logger.error("选择车辆失败，将在下次循环重试")
            return
        
        if self.selected_tank_ is not None:
            logger.info(f"当前车辆: {self.selected_tank_.name}")
        
        # 加入战斗
        if not self.enter_battle():
            logger.error("加入战斗失败，将在下次循环重试")
            return
        
        # 标记已处理，等待状态转换
        self.garage_handled_ = True
        logger.info("车库状态处理完成，等待进入战斗")
    
    def _handle_battle_state(self) -> None:
        """
        处理战斗状态：检测地图名称并启动导航AI
        """
        if self.battle_handled_:
            return

        self.end_handled_ = False
        self.garage_handled_ = False

        logger.info("检测到战斗状态，开始启动导航...")
        # 启动导航线程
        self.navigation_runtime_ = NavigationRuntime()
        self.navigation_thread_ = threading.Thread(target=self.navigation_runtime_.start, daemon=True)
        self.navigation_thread_.start()
        # 标记已处理
        self.battle_handled_ = True
        logger.info("战斗状态处理完成")
    
    def _handle_end_state(self) -> None:
        """
        处理结束状态：停止导航AI运行循环（保留初始化状态）
        """

        logger.info("检测到战斗结束状态，停止导航AI运行循环...")
        
        # 停止导航AI运行循环
        if self.navigation_runtime_ is not None:
            self.navigation_runtime_.stop()
        self.navigation_runtime_ = None

        if self.navigation_thread_ and self.navigation_thread_.is_alive():
            self.navigation_thread_.join(timeout=3.0)
        self.navigation_thread_ = None

        # 关闭结算页面并返回车库
        if not self.enter_garage():
            logger.error("返回车库失败，将在下次循环重试")
            return
        
        # 重置战斗状态标志，为下一局做准备
        self.battle_handled_ = False
        self.garage_handled_ = False
        logger.info("结束状态处理完成")
    
    def _handle_result_page_state(self) -> None:
        """
        处理结果页面状态：返回车库
        """
        logger.info("检测到结果页面状态，返回车库...")
        self.key_controller_.tap(Key.esc)
        logger.info("结果页面状态处理完成")

    
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
            time.sleep(self.selection_retry_interval_)
        
        logger.error("选择车辆超时")
        return False
    
    def _try_select_candidate(self, candidate: TankTemplate) -> bool:
        """尝试点击单个候选车辆"""
        success = self.template_matcher_.click_template(candidate.name, confidence=candidate.confidence)
        if success is not None:
            return True
        return False
    
    def enter_battle(self) -> bool:
        """
        点击加入战斗
        
        Returns:
            是否成功
        """
        logger.info("点击加入战斗...")
        
        success = self.template_matcher_.click_template("join_battle.png", confidence=0.85)
        if not success:
            logger.error("未找到加入战斗按钮")
            return False
        
        time.sleep(1.0)  # 等待进入战斗界面
        logger.info("已点击加入战斗")
        return True
    
    def enter_garage(self) -> bool:
        """
        返回车库
        
        Returns:
            是否成功
        """
        logger.info("退出结算界面，返回车库...")

        # 1. 奖励页面，按下esc键退出奖励页面
        success = self.template_matcher_.match_template("jie_suan_3.png", confidence=0.85)
        if success is not None:
            logger.info("在奖励页面，按下esc键退出结算界面")
            self.key_controller_.tap(Key.esc)
            return True
        
        # 2. 结算页面
        success = self.template_matcher_.match_template("jie_suan_2.png", confidence=0.85)
        if success is not None:
            logger.info("在结算页面，按下esc键退出结算界面")
            self.key_controller_.tap(Key.esc)
            return True
            
        # 2.当被击毁时，点击"返回车库"按钮
        # press esc
        success = self.template_matcher_.match_template("pingjia.png", confidence=0.85)
        if success is not None:
            logger.info("被击毁状态，需要按下 ESC 然后点击返回车库按钮")
            self.key_controller_.tap(Key.esc)
            time.sleep(3.0)
            success = self.template_matcher_.click_template("return_garage.png", confidence=0.85)
            if success is None:
                logger.error("未找到返回车库按钮")
                return False

            success = self.template_matcher_.click_template("leave_confirmation.png", confidence=0.85)
            if success is None:
                logger.error("未找到离开确认按钮")
                return False

        return True
    
    def is_running(self) -> bool:
        """
        检查事件驱动循环是否正在运行
        
        Returns:
            是否正在运行
        """
        return self.running_
    
    def __del__(self):
        """析构函数：确保资源清理"""
        self.stop()
