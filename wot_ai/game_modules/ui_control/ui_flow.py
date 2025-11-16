#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI 流程状态机模块

功能：
- 实现从车库到战斗的完整 UI 流程自动化
- 状态机管理流程状态
- 异常处理和重试机制
"""

from typing import Optional
from loguru import logger

from .actions import UIActions


class UIFlow:
    """UI 流程状态机"""
    
    def __init__(
        self,
        click_delay: float = 0.3,
        move_duration: float = 0.2,
        garage_timeout: float = 30.0,
        countdown_timeout: float = 20.0
    ):
        """
        初始化 UI 流程
        
        Args:
            click_delay: 点击后的延迟时间（秒）
            move_duration: 鼠标移动的持续时间（秒）
            garage_timeout: 等待车库界面就绪的超时时间（秒）
            countdown_timeout: 等待倒计时的超时时间（秒）
        """
        self.actions_ = UIActions(click_delay=click_delay, move_duration=move_duration)
        self.garage_timeout_ = garage_timeout
        self.countdown_timeout_ = countdown_timeout
    
    def StartBattle(self) -> bool:
        """
        执行从车库到战斗的完整流程
        
        Returns:
            是否成功完成流程
        """
        logger.info("开始执行 UI 流程：车库 → 战斗")
        
        try:
            # Step 1: 等待车库界面就绪
            if not self._WaitGarageReady():
                logger.error("等待车库界面失败")
                return False
            
            # Step 2: 选择坦克
            if not self._SelectTank():
                logger.error("选择坦克失败")
                return False
            
            # Step 3: 点击加入战斗
            if not self._ClickJoinBattle():
                logger.error("点击加入战斗失败")
                return False
            
            # Step 4: 等待倒计时
            if not self._WaitCountdown():
                logger.error("等待倒计时失败")
                return False
            
            # Step 5: 启动导航 AI（占位方法）
            self._LaunchNavigationAI()
            
            logger.info("UI 流程执行完成")
            return True
            
        except Exception as e:
            logger.error(f"UI 流程执行异常: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _WaitGarageReady(self) -> bool:
        """
        等待车库界面就绪（检测 join_battle.png 出现）
        
        Returns:
            是否成功检测到车库界面
        """
        logger.info("等待车库界面就绪...")
        
        # 使用较高的置信度检测静态按钮
        return self.actions_.WaitAppear(
            "join_battle.png",
            timeout=self.garage_timeout_,
            confidence=0.85
        )
    
    def _SelectTank(self) -> bool:
        """
        选择坦克（点击坦克卡片）
        
        Returns:
            是否成功点击坦克卡片
        """
        logger.info("选择坦克...")
        
        # 使用较高的置信度检测坦克卡片
        return self.actions_.ClickTemplate(
            "tank_card.png",
            timeout=5.0,
            confidence=0.85,
            max_retries=3
        )
    
    def _ClickJoinBattle(self) -> bool:
        """
        点击加入战斗按钮
        
        Returns:
            是否成功点击加入战斗按钮
        """
        logger.info("点击加入战斗...")
        
        # 使用较高的置信度检测按钮
        return self.actions_.ClickTemplate(
            "join_battle.png",
            timeout=5.0,
            confidence=0.85,
            max_retries=3
        )
    
    def _WaitCountdown(self) -> bool:
        """
        等待倒计时出现（检测 countdown_10.png）
        
        Returns:
            是否成功检测到倒计时
        """
        logger.info("等待倒计时...")
        
        # 倒计时数字可以使用较低的置信度
        return self.actions_.WaitAppear(
            "countdown_10.png",
            timeout=self.countdown_timeout_,
            confidence=0.7
        )
    
    def _LaunchNavigationAI(self) -> None:
        """
        启动导航 AI（暂时占位，后续实现）
        
        注意：此方法目前只是占位，实际集成需要根据具体需求实现
        """
        logger.info("导航 AI 启动接口（待实现）")
        # TODO: 后续实现导航 AI 的启动逻辑
        # 可能需要：
        # - 接收 NavigationMain 实例
        # - 调用 NavigationMain.Run()
        # - 或通过其他方式启动导航模块

