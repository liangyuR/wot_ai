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
import pyautogui

from src.core.global_context import GlobalContext
from src.utils.global_path import TemplatePath

def match_template(
    template_name: str,
    confidence: float = 0.85,
    region: Optional[Tuple[int, int, int, int]] = None,
    template_dir: Optional[str] = None
) -> Optional[Tuple[int, int]]:
    """
    在屏幕上查找模板，返回匹配位置的中心点坐标
    
    Args:
        template_name: 模板文件名（如 "join_battle.png"）
        confidence: 匹配置信度（0.0-1.0），默认 0.85
        region: 可选的搜索区域 (left, top, width, height)，None 表示全屏搜索
        template_dir: 自定义模板目录，默认使用程序目录下 templates
    
    Returns:
        匹配位置的中心点坐标 (x, y)，如果未找到则返回 None
    """
    # 获取模板路径
    template_path = TemplatePath(template_name)
    if template_path is None:
        return None
    
    try:
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

