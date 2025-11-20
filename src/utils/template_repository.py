#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模板仓库，根据分辨率自动读取对应的模板文件组
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger


class TemplateRepository:
    """负责扫描并缓存模板路径。"""

    SUPPORTED_SUFFIXES = (".png",)
    _instance: Optional['TemplateRepository'] = None

    def __new__(cls) -> 'TemplateRepository':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # 如果已经初始化过，跳过
        if hasattr(self, '_initialized'):
            return
        self.templates_: Dict[str, Path] = {}
        self.loaded_: bool = False
        self.search_dirs_: List[Path] = []
        self._initialized = True

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