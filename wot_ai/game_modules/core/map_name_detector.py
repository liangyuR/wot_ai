#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图名称识别模块

支持模板匹配和OCR两种方式识别地图名称。
"""

from typing import List, Optional
import numpy as np
import cv2
from loguru import logger
from difflib import SequenceMatcher

from wot_ai.game_modules.ui_control.matcher_pyautogui import match_template
import pytesseract
from wot_ai.utils.paths import get_project_root
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
        self.known_map_names_ = self._build_map_name_library()
    
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
    
    def detect_from_loading(self, frame: Optional[np.ndarray]) -> Optional[str]:
        """
        针对加载界面的地图识别逻辑，支持多个区域 OCR
        """
        if frame is None or not OCR_AVAILABLE:
            return None
        
        texts = self._extract_loading_texts(frame)
        return self._match_map_name_from_texts(texts)
    
    def detect_from_pause(self, frame: Optional[np.ndarray]) -> Optional[str]:
        """
        针对按住B键后的暂停界面识别逻辑
        """
        if frame is None or not OCR_AVAILABLE:
            return None
        
        texts = self._extract_loading_texts(frame)
        return self._match_map_name_from_texts(texts)
    
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

    def _extract_loading_texts(self, frame: np.ndarray) -> List[str]:
        """
        从加载界面的左半区和上半区提取 OCR 文本
        """
        h, w = frame.shape[:2]
        regions = [
            frame[:, :max(1, w // 2)],      # 左半部分
            frame[:max(1, h // 2), :]       # 上半部分
        ]
        texts: List[str] = []
        for region in regions:
            processed = self._preprocess_region(region)
            if processed is None:
                continue
            text = pytesseract.image_to_string(
                processed,
                lang="chi_sim+eng",
                config="--psm 6"
            )
            cleaned = self._clean_text(text)
            if cleaned:
                texts.append(cleaned)
        return texts

    def _preprocess_region(self, region: np.ndarray) -> Optional[np.ndarray]:
        """
        加载界面 OCR 预处理
        """
        if region.size == 0:
            return None
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8
        )
        inverted = cv2.bitwise_not(thresh)
        return inverted

    def _clean_text(self, text: str) -> str:
        """
        简单清洗 OCR 文本
        """
        if not text:
            return ""
        cleaned = text.replace("\n", "").replace(" ", "").strip()
        return cleaned

    def _match_known_map_name(self, text: str) -> Optional[str]:
        """
        将 OCR 文本与已知地图名称做模糊匹配
        """
        if not text:
            return None
        
        for name in self.known_map_names_:
            if name in text:
                return name
        
        best_name = None
        best_score = 0.0
        for name in self.known_map_names_:
            score = SequenceMatcher(None, name, text).ratio()
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name and best_score >= 0.65:
            logger.info(f"模糊匹配地图: {best_name} (score={best_score:.2f})")
            return best_name
        
        logger.debug(f"OCR 文本未匹配到地图: {text}")
        return None
    
    def _match_map_name_from_texts(self, texts: List[str]) -> Optional[str]:
        """
        遍历多段文本并返回首个匹配的地图名称
        """
        for text in texts:
            if not text:
                continue
            matched = self._match_known_map_name(text)
            if matched:
                return matched
        return None

    def _build_map_name_library(self) -> List[str]:
        """
        构建地图名称库，合并默认列表与 maps 目录下的文件名
        """
        names = set(self.common_map_names_)
        names.update(self._load_names_from_maps_dir())
        return sorted(names)

    def _load_names_from_maps_dir(self) -> List[str]:
        """
        从 maps 目录加载可用地图名称
        """
        try:
            maps_dir = get_project_root() / "maps"
        except Exception as exc:
            logger.warning(f"获取地图目录失败: {exc}")
            return []
        
        if not maps_dir.exists():
            return []
        
        candidates: List[str] = []
        for path in maps_dir.glob("*.png"):
            name = path.stem
            if name.endswith("_mask"):
                continue
            candidates.append(name)
        return candidates

