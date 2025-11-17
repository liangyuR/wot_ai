#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图名称识别模块

支持模板匹配和OCR两种方式识别地图名称。
"""

from typing import Optional
import numpy as np
import cv2
from loguru import logger

from wot_ai.game_modules.ui_control.matcher_pyautogui import match_template
import pytesseract
OCR_AVAILABLE = True


class MapNameDetector:
    """地图名称识别器"""
    
    def __init__(self, use_ocr: bool = True):
        """
        初始化地图名称识别器
        
        Args:
            use_ocr: 是否使用OCR作为备选方案（如果模板匹配失败）
        """
        self.use_ocr_ = use_ocr and OCR_AVAILABLE
        
        # 地图名称模板前缀
        self.template_prefix_ = "map_name_"
        
        # 常见地图名称列表（用于OCR识别后的匹配）
        self.common_map_names_ = [
            "胜利之门", "马利诺夫卡", "普罗霍洛夫卡", "鲁别克", "卡累利阿",
            "拉斯威利", "湖边的角逐", "荒蛮之地", "锡默尔斯多夫", "斯特拉特福",
            "费舍尔湾", "埃里-哈罗夫", "安斯克", "慕尼黑", "巴黎",
            "柏林", "明斯克", "哈尔科夫", "斯大林格勒", "库尔斯克"
        ]
    
    def detect(self, frame: Optional[np.ndarray] = None) -> Optional[str]:
        """
        识别地图名称
        
        Args:
            frame: 可选的屏幕截图（BGR格式），如果为None则自动截图
        
        Returns:
            地图名称字符串，如果识别失败则返回None
        """
        # 方法1: 尝试模板匹配
        map_name = self._detect_by_template()
        if map_name:
            logger.info(f"通过模板匹配识别地图: {map_name}")
            return map_name
        
        # 方法2: 如果模板匹配失败且启用OCR，尝试OCR识别
        if self.use_ocr_ and frame is not None:
            map_name = self._detect_by_ocr(frame)
            if map_name:
                logger.info(f"通过OCR识别地图: {map_name}")
                return map_name
        
        logger.warning("地图名称识别失败")
        return None
    
    def _detect_by_template(self) -> Optional[str]:
        """
        通过模板匹配识别地图名称
        
        Returns:
            地图名称，如果未找到则返回None
        """
        # 尝试匹配所有已知地图的模板
        for map_name in self.common_map_names_:
            template_name = f"{self.template_prefix_}{map_name}.png"
            center = match_template(template_name, confidence=0.75)
            if center is not None:
                return map_name
        
        return None
    
    def _detect_by_ocr(self, frame: np.ndarray) -> Optional[str]:
        """
        通过OCR识别地图名称
        
        Args:
            frame: 屏幕截图（BGR格式）
        
        Returns:
            地图名称，如果识别失败则返回None
        """
        if not OCR_AVAILABLE:
            return None
        
        try:
            # 假设地图名称显示在屏幕顶部中央区域
            # 这里需要根据实际游戏UI调整区域
            h, w = frame.shape[:2]
            region = frame[
                int(h * 0.05):int(h * 0.15),  # 顶部5%-15%
                int(w * 0.3):int(w * 0.7)      # 中央30%-70%
            ]
            
            # 预处理图像（提高OCR识别率）
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR识别
            text = pytesseract.image_to_string(binary, lang='chi_sim+eng')
            text = text.strip()
            
            if not text:
                return None
            
            # 在常见地图名称列表中查找匹配
            for map_name in self.common_map_names_:
                if map_name in text:
                    return map_name
            
            # 如果没有完全匹配，返回识别的文本（可能需要进一步处理）
            logger.debug(f"OCR识别文本: {text}，但未在常见地图列表中找到匹配")
            return text
            
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return None

