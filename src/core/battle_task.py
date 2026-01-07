#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
战斗任务模块

基于 hub state 的事件驱动模式，实现自动战斗循环。
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime
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
    
    # 常量
    SILVER_RESERVE_INTERVAL = 3600.0  # 1小时
    KEYBOARD_WAIT_AFTER_STOP = 0.5    # 停止导航后等待时间（秒）
    SETTLEMENT_TEMPLATE_CONFIDENCE = 0.85  # 结算页面模板匹配置信度
    
    def __init__(
        self,
        selection_retry_interval: float = 10.0,
        state_check_interval: float = 3,
        enable_silver_reserve: bool = False,
    ):
        """
        初始化战斗任务
        
        Args:
            selection_retry_interval: 选择失败后的重试间隔
            state_check_interval: 状态检测间隔（秒）
            enable_silver_reserve: 是否启用银币储备功能
        """
        self.tank_selector_ = TankSelector()
        self.state_machine_ = StateMachine()
        self.template_matcher_ = TemplateMatcher()
        self.map_name_detector_ = MapNameDetector()
        self.key_controller_ = KeyController()
        self.navigation_runtime_ = NavigationRuntime()
        
        config = GetGlobalConfig()
        self.game_restarter_ = GameRestarter(
            process_name=config.game.process_name,
            game_exe_path=config.game.exe_path,
            wait_seconds=config.game.restart_wait_seconds
        )
        self.stuck_timeout_ = config.game.stuck_timeout_seconds

        self.selection_retry_interval_ = selection_retry_interval
        self.state_check_interval_ = state_check_interval
        
        # 事件驱动相关
        self.running_ = False
        self.event_thread_: Optional[threading.Thread] = None
        
        # 选中的车辆
        self.selected_tank_: Optional[TankTemplate] = None
        
        # 状态监测
        self.last_state_ = GameState.UNKNOWN
        self.last_state_change_time_ = time.time()
        
        # 银币储备功能
        self._silver_reserve_enabled = enable_silver_reserve
        self._silver_reserve_interval = self.SILVER_RESERVE_INTERVAL
        self._silver_reserve_template = "silver_reserve.png"  # 模板名称
        self._last_silver_reserve_time: Optional[float] = None  # 上次激活时间
        
        # 自动停止配置（从配置文件读取）
        auto_stop_cfg = config.auto_stop
        self.auto_stop_ = auto_stop_cfg.enable
        self.run_hours_ = auto_stop_cfg.run_hours
        self.auto_shutdown_ = auto_stop_cfg.auto_shutdown
        self.start_time_: Optional[datetime] = None
        
        # 定时重启配置
        restart_cfg = config.scheduled_restart
        self.scheduled_restart_enabled_ = restart_cfg.enable
        self.scheduled_restart_hours_ = restart_cfg.interval_hours
    
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
        self.start_time_ = datetime.now()  # 记录启动时间
        
        logger.info(f"任务启动，运行时长限制: {self.run_hours_} 小时")
        if self.auto_stop_:
            logger.info(f"自动停止已启用，将在 {self.run_hours_} 小时后停止")
        if self.auto_shutdown_:
            logger.info("自动关机已启用，达到时长后将关机")
        if self.scheduled_restart_enabled_:
            logger.info(f"定时重启已启用，间隔: {self.scheduled_restart_hours_} 小时")
        
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
        
        if self.navigation_runtime_.is_running():
            self.navigation_runtime_.stop()
        
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
        
        try:
            while self.running_:
                try:
                    # 检查停止条件
                    if self._shouldStop():
                        logger.info("达到停止条件，退出循环")
                        break
                    
                    # 检查定时重启
                    if self._shouldScheduledRestart():
                        logger.info("达到定时重启时间，准备重启程序...")
                        self._restartProgram()
                        return  # execv 会替换进程，不会返回
                    
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
        finally:
            self.running_ = False
            logger.info("事件循环结束")
            
            # 如果配置了自动关机
            if self.auto_shutdown_ and self._shouldStop():
                self._shutdown()
    
    def _handle_garage_state(self) -> None:
        """
        处理车库状态：选择车辆并加入战斗
        """
        logger.info("检测到车库状态，开始选择车辆...")

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
        
        logger.info("车库状态处理完成，等待进入战斗")
    
    def _handle_battle_state(self) -> None:
        """
        处理战斗状态：检查银币储备 -> 识别地图 -> 启动导航AI
        """
        # 如果导航已经在运行，直接返回，避免重复处理
        if self.navigation_runtime_.is_running():
            return

        logger.info("检测到战斗状态，开始初始化...")

        # Step 1: 激活银币储备（如果启用）
        if self._shouldActivateSilverReserve():
            self._activateSilverReserve()
        else:
            logger.info("不需要激活银币储备")

        # Step 2: 识别地图名称（在启动导航之前）
        logger.info("识别地图名称...")
        self.key_controller_.press('b')
        time.sleep(2)
        map_name = self.map_name_detector_.detect()
        self.key_controller_.release('b')
        
        if not map_name:
            logger.error("地图识别失败，无法启动导航")
            return
        
        logger.info(f"地图识别成功: {map_name}")

        # Step 3: 使用已识别的地图名称启动导航
        if not self.navigation_runtime_.start(map_name=map_name):
            logger.error("导航启动失败")
            return
        
        logger.info("战斗状态处理完成，导航AI已启动")
    
    def _handle_end_state(self) -> None:
        """
        处理结束状态：停止导航AI运行循环（保留实例以便复用）
        """

        logger.info("检测到战斗结束状态，停止导航AI运行循环...")
        
        # 停止导航AI运行循环（保留实例以便下次复用）
        if self.navigation_runtime_.is_running():
            self.navigation_runtime_.stop()
            # 等待键盘操作完全停止，避免与后续按键操作冲突
            time.sleep(0.5)

        # 关闭结算页面并返回车库
        if not self.enter_garage():
            logger.error("返回车库失败，将在下次循环重试")
            return
        
        logger.info("结束状态处理完成")

    def _shouldActivateSilverReserve(self) -> bool:
        """检查是否需要激活银币储备"""
        return self._silver_reserve_enabled
        # if not self._silver_reserve_enabled:
        #     return False
        
        # if self._last_silver_reserve_time is None:
        #     return True  # 首次激活
        
        # return (time.time() - self._last_silver_reserve_time) >= self._silver_reserve_interval

    def _activateSilverReserve(self) -> bool:
        """激活银币储备"""
        logger.info("开始激活银币储备...")
        
        # 按住 B 键打开储备界面
        self.key_controller_.press('b')
        time.sleep(2)
        
        # 点击银币储备模板（最多2次）
        for attempt in range(2):
            if self.template_matcher_.click_template(
                self._silver_reserve_template, 
                confidence=self.SETTLEMENT_TEMPLATE_CONFIDENCE
            ):
                logger.info(f"银币储备模板点击成功（第 {attempt + 1} 次）")
                break
            logger.warning(f"银币储备模板点击失败（第 {attempt + 1} 次）")
            time.sleep(0.5)
        
        # 关闭储备界面
        self.key_controller_.release('b')
        
        self._last_silver_reserve_time = time.time()
        logger.info("银币储备激活完成，下次激活将在1小时后")
        return True
    
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
        return False
    
    def _try_select_candidate(self, candidate: TankTemplate) -> bool:
        """尝试点击单个候选车辆"""
        return self.template_matcher_.click_template(
            candidate.name, 
            confidence=candidate.confidence
        ) is not None
    
    def enter_battle(self) -> bool:
        """
        点击加入战斗
        
        Returns:
            是否成功
        """
        logger.info("点击加入战斗...")
        
        success = self.template_matcher_.click_template(
            "join_battle.png", 
            confidence=self.SETTLEMENT_TEMPLATE_CONFIDENCE
        )
        if not success:
            logger.error("未找到加入战斗按钮")
            return False
        
        time.sleep(1.0)  # 等待进入战斗界面
        logger.info("已点击加入战斗")
        return True
    
    def enter_garage(self) -> bool:
        """返回车库"""
        logger.info("退出结算界面，返回车库...")
        
        # 结算页面模板列表（按优先级）
        settlement_templates = [
            "jie_suan_1.png",
            "jie_suan_2.png",
            "jie_suan_3.png",
            "jie_suan_4.png",
            "jie_suan_5.png"
        ]
        
        # 检查并退出结算页面
        for template in settlement_templates:
            if self.template_matcher_.match_template(
                template, 
                confidence=self.SETTLEMENT_TEMPLATE_CONFIDENCE
            ):
                logger.info(f"检测到结算页面 ({template})，按下 ESC 退出")
                self.key_controller_.tap(Key.esc)
                return True
        
        # 处理被击毁状态
        if self.template_matcher_.match_template(
            "pingjia.png", 
            confidence=self.SETTLEMENT_TEMPLATE_CONFIDENCE
        ):
            return self._handle_destroyed_state()
        
        return True
    
    def _handle_destroyed_state(self) -> bool:
        """处理被击毁状态：返回车库"""
        logger.info("被击毁状态，按下 ESC 然后点击返回车库按钮")
        self.key_controller_.tap(Key.esc)
        time.sleep(3.0)
        
        if not self.template_matcher_.click_template(
            "return_garage.png", 
            confidence=self.SETTLEMENT_TEMPLATE_CONFIDENCE
        ):
            logger.error("未找到返回车库按钮")
            return False
        
        if not self.template_matcher_.click_template(
            "leave_confirmation.png", 
            confidence=self.SETTLEMENT_TEMPLATE_CONFIDENCE
        ):
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
    
    def _shouldStop(self) -> bool:
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
    
    def _shouldScheduledRestart(self) -> bool:
        """
        检查是否应该执行定时重启
        
        Returns:
            是否应该重启
        """
        if not self.scheduled_restart_enabled_:
            return False
        
        if self.start_time_ is None:
            return False
        
        elapsed = datetime.now() - self.start_time_
        elapsed_hours = elapsed.total_seconds() / 3600.0
        
        return elapsed_hours >= self.scheduled_restart_hours_
    
    def _restartProgram(self) -> None:
        """重启程序（使用 os.execv 替换当前进程）"""
        logger.info("正在重启程序...")

        python = sys.executable
        script = sys.argv[0]
        args = sys.argv[1:]

        # 添加 --auto-start 参数以便重启后自动启动
        if "--auto-start" not in args:
            args.append("--auto-start")

        logger.info(f"重启命令: {python} {script} {' '.join(args)}")

        try:
            os.execv(python, [python, script] + args)
        except Exception as e:
            logger.error(f"重启失败: {e}")
            # 如果 execv 失败，尝试使用 subprocess
            try:
                subprocess.Popen([python, script] + args)
                sys.exit(0)
            except Exception as e2:
                logger.error(f"备用重启也失败: {e2}")
    
    def _shutdown(self) -> None:
        """执行关机操作（需要管理员权限）"""
        logger.info("执行自动关机...")
        
        try:
            subprocess.run(['shutdown', '/s', '/t', '10'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"关机命令执行失败: {e}")
        except Exception as e:
            logger.error(f"关机操作异常: {e}")

    def __del__(self):
        """析构函数：确保资源清理"""
        self.stop()
