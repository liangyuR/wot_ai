#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模板匹配模块：使用 PyAutoGUI 进行模板匹配

功能：
- 使用 PyAutoGUI 的 locateOnScreen API 进行模板匹配
- 返回匹配位置的中心点坐标
- 支持区域限制和置信度调整
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

try:
    import pyautogui
except ImportError:
    logger.error("pyautogui 未安装，请运行: pip install pyautogui")
    pyautogui = None


# 模板目录路径（相对于模块目录）
_TEMPLATE_DIR = Path(__file__).parent / "template_repo"


def _GetTemplatePath(template_name: str) -> Optional[Path]:
    """
    获取模板文件的完整路径
    
    Args:
        template_name: 模板文件名（如 "join_battle.png"）
    
    Returns:
        模板文件的 Path 对象，如果不存在则返回 None
    """
    template_path = _TEMPLATE_DIR / template_name
    if not template_path.exists():
        logger.warning(f"模板文件不存在: {template_path}")
        return None
    return template_path


def match_template(
    template_name: str,
    confidence: float = 0.85,
    region: Optional[Tuple[int, int, int, int]] = None
) -> Optional[Tuple[int, int]]:
    """
    在屏幕上查找模板，返回匹配位置的中心点坐标
    
    Args:
        template_name: 模板文件名（如 "join_battle.png"）
        confidence: 匹配置信度（0.0-1.0），默认 0.85
        region: 可选的搜索区域 (left, top, width, height)，None 表示全屏搜索
    
    Returns:
        匹配位置的中心点坐标 (x, y)，如果未找到则返回 None
    """
    if pyautogui is None:
        logger.error("pyautogui 未安装，无法进行模板匹配")
        return None
    
    # 获取模板路径
    template_path = _GetTemplatePath(template_name)
    if template_path is None:
        return None
    
    try:
        # 使用 PyAutoGUI 的 locateOnScreen 进行匹配
        # region 参数格式: (left, top, width, height)
        location = pyautogui.locateOnScreen(
            str(template_path),
            confidence=confidence,
            region=region
        )
        
        if location is None:
            logger.debug(f"未找到模板: {template_name}")
            return None
        
        # 获取匹配区域的中心点
        center = pyautogui.center(location)
        logger.debug(f"找到模板 {template_name} 在位置: {center}")
        return center
        
    except pyautogui.ImageNotFoundException:
        logger.debug(f"未找到模板: {template_name}")
        return None
    except Exception as e:
        logger.error(f"模板匹配失败 {template_name}: {e}")
        return None

