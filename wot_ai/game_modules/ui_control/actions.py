#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI 动作封装模块

功能：
- 封装鼠标移动和点击操作
- 等待元素出现/消失
- 提供平滑移动和延迟控制
"""

import time
from typing import Optional, Tuple
from loguru import logger

try:
    import pyautogui
    # 启用安全模式：鼠标移到屏幕角落可中断
    pyautogui.FAILSAFE = True
except ImportError:
    logger.error("pyautogui 未安装，请运行: pip install pyautogui")
    pyautogui = None

from .matcher_pyautogui import match_template


class UIActions:
    """UI 动作封装类"""
    
    def __init__(self, click_delay: float = 0.3, move_duration: float = 0.2):
        """
        初始化 UI 动作类
        
        Args:
            click_delay: 点击后的延迟时间（秒），让游戏 UI 有时间响应
            move_duration: 鼠标移动的持续时间（秒），用于平滑移动
        """
        if pyautogui is None:
            raise RuntimeError("pyautogui 未安装，无法使用 UIActions")
        
        self.click_delay_ = click_delay
        self.move_duration_ = move_duration
    
    def MoveTo(self, x: int, y: int, duration: Optional[float] = None) -> None:
        """
        移动鼠标到指定位置
        
        Args:
            x: 目标 X 坐标
            y: 目标 Y 坐标
            duration: 移动持续时间（秒），None 则使用默认值
        """
        if duration is None:
            duration = self.move_duration_
        
        try:
            pyautogui.moveTo(x, y, duration=duration)
            logger.debug(f"鼠标移动到: ({x}, {y})")
        except Exception as e:
            logger.error(f"鼠标移动失败 ({x}, {y}): {e}")
            raise
    
    def Click(self, x: int, y: int) -> bool:
        """
        在指定位置点击鼠标
        
        Args:
            x: 点击 X 坐标
            y: 点击 Y 坐标
        
        Returns:
            是否成功
        """
        try:
            pyautogui.click(x, y)
            logger.info(f"点击位置: ({x}, {y})")
            time.sleep(self.click_delay_)
            return True
        except Exception as e:
            logger.error(f"点击失败 ({x}, {y}): {e}")
            return False
    
    def ClickTemplate(
        self,
        template_name: str,
        timeout: float = 5.0,
        confidence: float = 0.85,
        region: Optional[Tuple[int, int, int, int]] = None,
        max_retries: int = 3
    ) -> bool:
        """
        查找模板并点击其中心点
        
        Args:
            template_name: 模板文件名
            timeout: 超时时间（秒）
            confidence: 匹配置信度
            region: 可选的搜索区域
            max_retries: 最大重试次数
        
        Returns:
            是否成功找到并点击
        """
        start_time = time.time()
        retries = 0
        
        while retries < max_retries:
            # 检查超时
            if time.time() - start_time > timeout:
                logger.warning(f"点击模板超时: {template_name} (timeout={timeout}s)")
                return False
            
            # 查找模板
            center = match_template(template_name, confidence=confidence, region=region)
            
            if center is not None:
                # 找到模板，点击中心点
                return self.Click(center[0], center[1])
            
            # 未找到，等待一小段时间后重试
            time.sleep(0.2)
            retries += 1
        
        logger.warning(f"未找到模板，点击失败: {template_name} (重试 {retries} 次)")
        return False
    
    def WaitAppear(
        self,
        template_name: str,
        timeout: float = 30.0,
        confidence: float = 0.85,
        region: Optional[Tuple[int, int, int, int]] = None,
        check_interval: float = 0.5
    ) -> bool:
        """
        等待模板出现在屏幕上
        
        Args:
            template_name: 模板文件名
            timeout: 超时时间（秒）
            confidence: 匹配置信度
            region: 可选的搜索区域
            check_interval: 检查间隔（秒）
        
        Returns:
            是否在超时前找到模板
        """
        start_time = time.time()
        
        logger.info(f"等待模板出现: {template_name} (timeout={timeout}s)")
        
        while True:
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"等待模板出现超时: {template_name} (timeout={timeout}s)")
                return False
            
            # 查找模板
            center = match_template(template_name, confidence=confidence, region=region)
            
            if center is not None:
                logger.info(f"模板已出现: {template_name} 在位置 {center}")
                return True
            
            # 未找到，等待后继续检查
            time.sleep(check_interval)
    
    def WaitDisappear(
        self,
        template_name: str,
        timeout: float = 30.0,
        confidence: float = 0.85,
        region: Optional[Tuple[int, int, int, int]] = None,
        check_interval: float = 0.5
    ) -> bool:
        """
        等待模板从屏幕上消失
        
        Args:
            template_name: 模板文件名
            timeout: 超时时间（秒）
            confidence: 匹配置信度
            region: 可选的搜索区域
            check_interval: 检查间隔（秒）
        
        Returns:
            是否在超时前模板消失
        """
        start_time = time.time()
        
        logger.info(f"等待模板消失: {template_name} (timeout={timeout}s)")
        
        # 首先确认模板存在
        initial_check = match_template(template_name, confidence=confidence, region=region)
        if initial_check is None:
            logger.info(f"模板已不存在: {template_name}")
            return True
        
        while True:
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"等待模板消失超时: {template_name} (timeout={timeout}s)")
                return False
            
            # 查找模板
            center = match_template(template_name, confidence=confidence, region=region)
            
            if center is None:
                logger.info(f"模板已消失: {template_name}")
                return True
            
            # 仍然存在，等待后继续检查
            time.sleep(check_interval)

