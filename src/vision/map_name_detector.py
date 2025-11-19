#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图名称识别模块（模板匹配版）

通过 `match_template` 在屏幕上查找地图标题模板，完全替代 OCR。
"""

from typing import Dict, List, Optional
from pathlib import Path

from loguru import logger



class MapNameDetector:
    """基于模板匹配的地图名称识别器"""

    def __init__(self) -> None:
        self.known_map_names_ = self._load_names_from_maps_dir()
        self.map_names = self.known_map_names_
        self.template_confidence_ = 0.82
        self.template_mapping_ = self._load_map_templates()
        if not self.template_mapping_:
            logger.warning("未加载到地图标题模板，将无法识别地图名称")

    def detect(self, frame=None) -> Optional[str]:
        """
        使用模板匹配识别地图名称。

        Args:
            frame: 兼容旧版接口的占位参数（当前实现未使用）。
        """
        from src.ui_control.matcher_pyautogui import match_template

        if not self.template_mapping_:
            return None
        for map_name, template_files in self.template_mapping_.items():
            for template_name in template_files:
                try:
                    center = match_template(
                        template_name,
                        confidence=self.template_confidence_,
                        region=None  # 需求：全屏范围搜索
                    )
                except Exception as exc:
                    logger.error(f"模板匹配异常 {template_name}: {exc}")
                    continue

                if center:
                    logger.info(f"模板匹配成功: {map_name} (模板: {template_name}, 坐标: {center})")
                    return map_name

        logger.warning("模板匹配未识别到地图名称")
        return None

    def _load_map_templates(self) -> Dict[str, List[str]]:
        """
        加载地图标题模板。

        目录结构：resource/template/<tier>/map_names/*.png
        命名约定：{map_name}.png 或 {map_name}__variant.png
        """
        from src.utils.global_path import GetMapTemplatePath
        template_dir = Path(GetMapTemplatePath())
        if not template_dir.exists():
            logger.error(f"地图标题模板目录不存在: {template_dir}")
            return {}

        mapping: Dict[str, List[str]] = {}
        for tpl_path in sorted(template_dir.glob("*.png")):
            map_name = tpl_path.stem.split("__")[0]
            template_rel = tpl_path.name
            mapping.setdefault(map_name, []).append(template_rel)

        total_templates = sum(len(files) for files in mapping.values())
        logger.info(f"已加载 {total_templates} 个地图标题模板，覆盖 {len(mapping)} 张地图")
        return mapping

    def _load_names_from_maps_dir(self) -> List[str]:
        """
        从 maps 目录加载可用地图名称（保留外部兼容）。
        """
        from src.utils.global_path import GetMapsDir
        maps_dir = GetMapsDir()
        if not maps_dir.exists():
            logger.error(f"地图目录不存在: {maps_dir}")
            return []
        return [path.stem for path in maps_dir.glob("*.png") if not path.stem.endswith("_mask")]


if __name__ == "__main__":
    """用于测试地图名称检测功能"""
    import argparse
    import sys
    import cv2

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
    # 输出结果
    if map_name:
        print(f"检测到地图名称: {map_name}")
        sys.exit(0)
    else:
        print("未检测到地图名称")
