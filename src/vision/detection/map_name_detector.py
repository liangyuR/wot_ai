#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图名称识别模块

使用OCR方式识别地图名称。通过按住B键打开菜单界面，截图后识别顶部区域的地图信息。
"""

from typing import List, Optional
import argparse
import sys
import re
from pathlib import Path
import numpy as np
import cv2
from loguru import logger
from rapidfuzz import fuzz, process

import pytesseract
from pynput import keyboard
from wot_ai.game_modules.core.actions import screenshot_with_key_hold
from wot_ai.utils.paths import get_project_root

class MapNameDetector:
    """地图名称识别器（使用OCR方式）"""
    
    def __init__(self):
        """
        初始化地图名称识别器
        """
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

        Returns:
            地图名称字符串，如果识别失败则返回None
        """

        # 如果未提供截图，则自动按住B键截图
        if frame is None:
            frame = screenshot_with_key_hold(keyboard.KeyCode.from_char('b'), hold_duration=4.0, warmup=2)
            if frame is None:
                logger.error("按住B键后截图失败")
                return None
    
        # 从顶部1/4区域提取第一行文本并识别地图名称
        map_name = self._extract_top_text(frame)
        if map_name:
            logger.info(f"通过OCR识别地图: {map_name}")
            return map_name
        
        logger.warning("地图名称识别失败")
        return None
    
    def _extract_top_text(self, frame: np.ndarray) -> Optional[str]:
        """
        从截图顶部1/4区域提取第一行文本并匹配地图名称
        
        Args:
            frame: 屏幕截图（BGR格式）
        
        Returns:
            地图名称，如果识别失败则返回None
        """
        try:
            h, w = frame.shape[:2]
            left_top_region = frame[0:h//4, w//4:w//2]
            
            if left_top_region.size == 0:
                logger.warning("左侧顶部区域为空")
                return None
            
            # cv2.imshow("Processed", left_top_region)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # OCR识别，尝试多种配置以提高准确率
            text = pytesseract.image_to_string(
                left_top_region,
                lang="chi_sim",
                config="--psm 6"
            )
            if not text:
                logger.debug("未识别到文本")
                return None
            
            # 提取所有非空文本行并清理
            text_lines = []
            for line in text.split('\n'):
                cleaned = self._clean_ocr_text(line)
                if cleaned:
                    text_lines.append(cleaned)
            
            if not text_lines:
                logger.debug("未识别到有效文本")
                return None
            
            logger.debug(f"OCR识别文本（清理后）: {text_lines}")
            
            map_name = self._match_map_name_from_texts(text_lines)
            return map_name
            
        except Exception as e:
            logger.error(f"提取顶部文本失败: {e}")
            return None

    def _clean_ocr_text(self, text: str) -> str:
        """
        清理OCR识别的文本，提取中文字符片段
        
        Args:
            text: 原始OCR文本
        
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除常见噪音字符，但保留中文字符、数字和常见标点
        # 提取中文字符、数字和常见标点
        cleaned = re.sub(r'[^\u4e00-\u9fa5\d，。、：；！？·\s]', '', text)
        # 移除多余空格
        cleaned = re.sub(r'\s+', '', cleaned)
        # 移除纯数字行（可能是噪音）
        if cleaned and cleaned.isdigit():
            return ""
        
        return cleaned.strip()
    
    def _clean_text(self, text: str) -> str:
        """
        简单清洗 OCR 文本（保留用于兼容性）
        """
        return self._clean_ocr_text(text)

    def _match_known_map_name(self, text: str) -> Optional[str]:
        """
        将 OCR 文本与已知地图名称做模糊匹配
        
        使用 rapidfuzz/fuzzywuzzy 库进行字符串相似度匹配，阈值50%
        """
        if not text:
            return None
        
        # 策略1: 完全包含匹配（快速路径）
        for name in self.known_map_names_:
            if name in text:
                logger.info(f"完全匹配地图: {name} (文本: {text})")
                return name
        
        # 策略2: 提取中文字符片段
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]+', text)
        candidates = [text]
        if chinese_chars:
            candidates.extend(chinese_chars)
            candidates.append(''.join(chinese_chars))
        
        # 策略3: 使用 rapidfuzz 进行模糊匹配
        best_match = None
        best_score = 0.0
        best_candidate = ""
        
        for candidate_text in candidates:
            if len(candidate_text) < 2:
                continue
            
            # 使用 process.extractOne 自动找最佳匹配
            result = process.extractOne(
                candidate_text,
                self.known_map_names_,
                scorer=fuzz.partial_ratio,  # 部分匹配，适合子串情况
                score_cutoff=50  # 50%阈值
            )
            if result:
                matched_name, score, _ = result
                if score > best_score:
                    best_score = score
                    best_match = matched_name
                    best_candidate = candidate_text
        
        # 如果部分匹配没找到，尝试完整匹配
        if best_score < 50:
            for candidate_text in candidates:
                if len(candidate_text) < 2:
                    continue
                
                result = process.extractOne(
                    candidate_text,
                    self.known_map_names_,
                    scorer=fuzz.ratio,  # 完整匹配
                    score_cutoff=50
                )
                if result:
                    matched_name, score, _ = result
                    if score > best_score:
                        best_score = score
                        best_match = matched_name
                        best_candidate = candidate_text
        
        if best_match and best_score >= 50:
            logger.info(f"模糊匹配地图: {best_match} (score={best_score:.1f}, 匹配文本: {best_candidate}, 原始文本: {text})")
            return best_match
        
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


def main():
    """主函数，用于测试地图名称检测功能"""
    parser = argparse.ArgumentParser(
        description="测试地图名称检测功能（使用OCR方式）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="图片文件路径（按住B键后的菜单界面截图）"
    )
    
    args = parser.parse_args()
    
    # 检查图片文件是否存在
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"图片文件不存在: {image_path}")
        sys.exit(1)
    
    # 读取图片
    try:
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.error(f"无法读取图片: {image_path}")
            sys.exit(1)
        logger.info(f"成功读取图片: {image_path}, 尺寸: {frame.shape}")
    except Exception as e:
        logger.error(f"读取图片失败: {e}")
        sys.exit(1)
    
    # 创建检测器
    detector = MapNameDetector()
    
    # 使用提供的截图进行检测
    logger.info("开始识别地图名称...")
    map_name = detector.detect(frame)
    w
    # 输出结果
    if map_name:
        print(f"检测到地图名称: {map_name}")
        sys.exit(0)
    else:
        print("未检测到地图名称")


if __name__ == "__main__":
    main()
