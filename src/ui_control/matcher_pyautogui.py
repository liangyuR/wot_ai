#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模板匹配模块：使用 PyAutoGUI 进行模板匹配，并缓存模板路径。
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger
import pyautogui


class TemplateRepository:
    """负责扫描并缓存模板路径。"""

    SUPPORTED_SUFFIXES = (".png",)

    def __init__(self) -> None:
        self.templates_: Dict[str, Path] = {}
        self.loaded_: bool = False
        self.search_dirs_: List[Path] = []

    def ensure_loaded(self, force_reload: bool = False) -> None:
        if force_reload or not self.loaded_:
            self._load_templates()

    def get_path(self, template: Union[str, Path]) -> Optional[Path]:
        tpl_path = Path(template)
        if tpl_path.is_absolute():
            return tpl_path if tpl_path.exists() else None

        norm_key = self._normalize_key(template)
        return self.templates_.get(norm_key)

    def find_on_disk(self, template: Union[str, Path]) -> Optional[Path]:
        """按当前搜索目录尝试定位单个模板（用于兜底）。"""
        tpl = Path(template)
        for base_dir in self.search_dirs_:
            candidate = base_dir / tpl
            if candidate.exists():
                return candidate
        return None

    def register_path(self, path: Path) -> None:
        """将磁盘路径加入缓存（用于兜底命中后复用）。"""
        path = path.resolve()
        for base_dir in self.search_dirs_:
            try:
                rel = path.relative_to(base_dir).as_posix()
                self._add_key(rel, path)
                break
            except ValueError:
                continue
        self._add_key(path.name, path)

    def _load_templates(self) -> None:
        start = time.time()
        self.templates_.clear()
        dirs = self._build_search_dirs()
        self.search_dirs_ = dirs
        total = 0

        for base_dir in dirs:
            if not base_dir.exists():
                logger.debug(f"模板目录不存在，跳过: {base_dir}")
                continue
            for file in base_dir.rglob("*"):
                if file.is_file() and file.suffix.lower() in self.SUPPORTED_SUFFIXES:
                    self._register_file(base_dir, file)
                    total += 1

        elapsed = (time.time() - start) * 1000.0
        logger.debug(
            f"模板仓库加载完成，共 {total} 个模板，耗时 {elapsed:.1f}ms"
        )
        self.loaded_ = True

    def _register_file(self, base_dir: Path, file_path: Path) -> None:
        try:
            rel_key = file_path.relative_to(base_dir).as_posix()
            self._add_key(rel_key, file_path)
        except ValueError:
            pass
        self._add_key(file_path.name, file_path)
        self._add_key(file_path.stem, file_path)

    def _add_key(self, key: str, path: Path) -> None:
        norm_key = self._normalize_key(key)
        if not norm_key:
            return
        existing = self.templates_.get(norm_key)
        if existing and existing != path:
            logger.warning(
                f"模板键冲突: {key} 已指向 {existing}，忽略 {path}"
            )
            return
        self.templates_[norm_key] = path

    def _build_search_dirs(self) -> List[Path]:
        from src.utils.global_path import GetHubTemplatePath, GetMapTemplatePath, GetVehicleScreenshotsDir
        dirs = [
            Path(GetHubTemplatePath()),
            Path(GetMapTemplatePath()),
            GetVehicleScreenshotsDir(),
        ]

        result = []
        seen = set()
        for directory in dirs:
            if not directory:
                continue
            resolved = Path(directory).resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            result.append(resolved)
        return result

    @staticmethod
    def _normalize_key(value: Union[str, Path]) -> str:
        if value is None:
            return ""
        key = str(value).replace("\\", "/").strip()
        return key.lower()


_TEMPLATE_REPOSITORY = TemplateRepository()


def match_template(
    template: Union[str, Path],
    confidence: float = 0.85,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """
    在屏幕上查找模板，返回匹配位置的中心点坐标。

    Args:
        template: 模板名称、相对路径或绝对路径
        confidence: 匹配置信度（0.0-1.0），默认 0.85
        region: 搜索区域 (left, top, width, height)，None 表示全屏
    """
    tpl_path = Path(template)
    template_path: Optional[Path] = None

    if tpl_path.is_absolute():
        if tpl_path.exists():
            template_path = tpl_path
        else:
            logger.error(f"模板不存在: {tpl_path}")
            return None
    else:
        _TEMPLATE_REPOSITORY.ensure_loaded()
        template_path = _TEMPLATE_REPOSITORY.get_path(template)
        if template_path is None:
            fallback = _TEMPLATE_REPOSITORY.find_on_disk(template)
            if fallback:
                _TEMPLATE_REPOSITORY.register_path(fallback)
                template_path = fallback

    if template_path is None:
        logger.debug("模板未找到: %s", template)
        return None

    try:
        location = pyautogui.locateOnScreen(
            str(template_path),
            confidence=confidence,
            region=region,
        )
        if location:
            center = pyautogui.center(location)
            logger.debug(f"找到模板 {template_path} 在位置: {center}")
            return center
        logger.debug(f"未在屏幕上找到模板: {template_path}")
        return None
    except Exception as exc:
        logger.error(f"模板匹配失败 {template_path}: {exc}")
        return None
