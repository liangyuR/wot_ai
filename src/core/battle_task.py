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
from src.core.ai_controller import AIController
from src.core.tank_selector import TankSelector, TankTemplate
from src.ui_control.actions import UIActions
from src.navigation.config.models import NavigationConfig
from src.vision.detection.map_name_detector import MapNameDetector
from src.listeners.global_listener import GlobalInputListener
from src.navigation.service.control_service import ControlService


class BattleTask:
    """战斗任务（事件驱动模式）"""
    
    def __init__(
        self,
        tank_selector: TankSelector,
        state_machine: StateMachine,
        map_detector: MapNameDetector,
        ai_config: NavigationConfig,
        ui_actions: UIActions,
        selection_retry_interval: float = 10.0,
        selection_timeout: float = 120.0,
        state_check_interval: float = 5
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
            state_check_interval: 状态检测间隔（秒）
        """
        self.tank_selector_ = tank_selector
        self.state_machine_ = state_machine
        self.map_detector_ = map_detector
        self.ai_config_ = ai_config
        self.ai_controller_ = AIController()
        self.ui_actions_ = ui_actions
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
        
        # 热键监听器
        self.input_listener_: Optional[GlobalInputListener] = None
        try:
            self.input_listener_ = GlobalInputListener()
            self.input_listener_.SetHotkeyCallback("f9", self._on_f9_pressed)
            self.input_listener_.SetHotkeyCallback("f10", self._on_f10_pressed)
            self.input_listener_.Start()
            logger.info("热键监听器已启动 (F9: 启动, F10: 停止)")
        except Exception as e:
            logger.warning(f"热键监听器启动失败: {e}")
    
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
        if self.ai_controller_.is_running():
            logger.info("停止导航AI...")
            self.ai_controller_.stop()
        
        # 等待事件循环线程结束
        if self.event_thread_ and self.event_thread_.is_alive():
            self.event_thread_.join(timeout=3.0)
        
        logger.info("事件驱动循环已停止")
    
    def _event_loop(self) -> None:
        """
        事件驱动主循环
        """
        logger.info("事件循环开始运行")
        
        while self.running_:
            try:
                # 更新状态机
                self.state_machine_.update()
                current_state = self.state_machine_.current_state()
                
                # 根据当前状态执行相应操作
                if current_state == GameState.IN_GARAGE:
                    self._handle_garage_state()
                elif current_state == GameState.IN_BATTLE:
                    self._handle_battle_state()
                elif current_state == GameState.IN_END:
                    self._handle_end_state()
                elif current_state == GameState.IN_RESULT_PAGE:
                    self._handle_result_page_state()
                elif current_state == GameState.UNKNOWN:
                    # 未知状态，不做处理
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
        
        # 如果导航AI未初始化，在第一次进入车库时初始化（使用默认地图）
        if not self.ai_controller_.is_initialized():
            logger.info("导航AI未初始化，在车库状态进行初始化...")
            if not self.ai_controller_.start(self.ai_config_, "default"):
                logger.error("导航AI初始化失败")
                return
            logger.info("导航AI初始化完成（YOLO模型已加载）")
        
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
        
        logger.info("检测到战斗状态，开始识别地图并启动导航AI...")
        
        # 等待一段时间让游戏稳定
        time.sleep(10.0)
        
        # 识别地图名称
        map_name = self.map_detector_.detect()
        if not map_name:
            logger.warning("地图名称识别失败，使用默认地图")
            map_name = "default"
        else:
            logger.info(f"识别到地图: {map_name}")
        
        # 如果导航AI已初始化，更新掩码
        if self.ai_controller_.is_initialized():
            logger.info("导航AI已初始化，更新掩码...")
            if not self.ai_controller_.update_mask(map_name):
                logger.warning("掩码更新失败，继续使用当前掩码")
            else:
                logger.info("掩码更新成功")
        
        # 启动导航AI运行循环
        if not self.ai_controller_.is_running():
            if not self.ai_controller_.is_initialized():
                # 如果未初始化，执行完整初始化
                if not self.ai_controller_.start(self.ai_config_, map_name):
                    logger.error("导航AI启动失败")
                    return
                logger.info("导航AI已启动")
            else:
                # 已初始化，只启动运行循环
                if not self.ai_controller_.start(self.ai_config_, map_name):
                    logger.error("导航AI运行循环启动失败")
                    return
                logger.info("导航AI运行循环已启动")
        else:
            logger.debug("导航AI已在运行")
        
        # 标记已处理
        self.battle_handled_ = True
        logger.info("战斗状态处理完成")
    
    def _handle_end_state(self) -> None:
        """
        处理结束状态：停止导航AI运行循环（保留初始化状态）
        """
        if self.end_handled_:
            return
        
        logger.info("检测到战斗结束状态，停止导航AI运行循环...")
        
        # 停止导航AI运行循环（保留初始化状态）
        if self.ai_controller_.is_running():
            self.ai_controller_.stop()
            logger.info("导航AI运行循环已停止（保留初始化状态）")
        
        # 重置战斗状态标志，为下一局做准备
        self.battle_handled_ = False

        # 关闭结算页面并返回车库
        if not self.enter_garage():
            logger.error("返回车库失败，将在下次循环重试")
            return
        
        # 标记已处理
        self.end_handled_ = True
        logger.info("结束状态处理完成")
    
    def _handle_result_page_state(self) -> None:
        """
        处理结算页面状态：关闭结算页面并返回车库
        """
        if self.result_page_handled_:
            return
        
        logger.info("检测到结算页面状态，开始返回车库...")
        
        # 关闭结算页面并返回车库
        if not self.enter_garage():
            logger.error("返回车库失败，将在下次循环重试")
            return
        
        # 重置所有状态标志，准备下一局
        self.garage_handled_ = False
        self.battle_handled_ = False
        self.end_handled_ = False
        self.result_page_handled_ = False
        
        # 标记已处理
        self.result_page_handled_ = True
        logger.info("结算页面状态处理完成，已返回车库")
    
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
        success = self.ui_actions_.SelectVehicle(
            template_name=candidate.name,
            template_dir=str(candidate.directory),
            confidence=candidate.confidence
        )
        if success:
            time.sleep(0.5)
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
        
        time.sleep(1.0)  # 等待进入战斗界面
        logger.info("已点击加入战斗")
        return True
    
    def enter_garage(self) -> bool:
        """
        退出结算界面，返回车库
        
        Returns:
            是否成功
        """
        logger.info("退出结算界面，返回车库...")

        # 1. 当前在结算页面
        success = self.ui_actions_.WaitAppear(
            "space_jump.png",
            timeout=5.0,
            confidence=0.85,
            max_retries=3
        )
        if success:
            ControlService().TapKey("esc")
            return True
        
        # 2.当被击毁时，点击"返回车库"按钮
        # press esc
        ControlService().TapKey("esc")

        # 点击"返回车库"按钮
        success = self.ui_actions_.ClickTemplate(
            "return_garage.png",
            timeout=5.0,
            confidence=0.85,
            max_retries=3
        )
        if not success:
            logger.error("未找到返回车库按钮")
            return False
        
        time.sleep(5.0)  # 等待返回车库

        logger.info("已点击返回车库按钮")
        return True
    
    def _on_f9_pressed(self) -> None:
        """F9 热键回调：启动事件驱动"""
        if not self.running_:
            logger.info("F9 热键：启动事件驱动循环")
            self.start()
        else:
            logger.info("F9 热键：事件驱动循环已在运行")
    
    def _on_f10_pressed(self) -> None:
        """F10 热键回调：停止事件驱动和导航AI"""
        if self.running_:
            logger.info("F10 热键：停止事件驱动循环")
            self.stop()
        else:
            logger.info("F10 热键：事件驱动循环未运行")
    
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
        if self.input_listener_:
            self.input_listener_.Stop()
