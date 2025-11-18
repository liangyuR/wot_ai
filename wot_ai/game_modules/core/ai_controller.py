#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI控制器模块

封装 NavigationMain 为战斗中的AI控制器，根据地图名称自动加载对应的mask掩码。
"""

import threading
from typing import Optional, TYPE_CHECKING
from loguru import logger

from wot_ai.config import get_program_dir
from wot_ai.game_modules.navigation.config.models import NavigationConfig
from wot_ai.game_modules.core.global_context import GlobalContext

if TYPE_CHECKING:
    from wot_ai.game_modules.navigation.navigation_main import NavigationMain

# 延迟导入NavigationMain以避免循环导入
def _get_navigation_main():
    from wot_ai.game_modules.navigation.navigation_main import NavigationMain
    return NavigationMain


class AIController:
    """AI控制器"""
    
    def __init__(self):
        """初始化AI控制器"""
        self.nav_main_: Optional['NavigationMain'] = None
        self.config_: Optional[NavigationConfig] = None
        self.map_name_: Optional[str] = None
        self.running_ = False
        self.thread_: Optional[threading.Thread] = None
    
    def start(self, config: NavigationConfig, map_name: str) -> bool:
        """
        启动AI控制器
        
        Args:
            config: NavigationConfig配置对象
            map_name: 地图名称，用于加载对应的mask掩码
        
        Returns:
            是否成功启动
        """
        if self.running_:
            logger.warning("AI控制器已在运行")
            return False
        
        self.map_name_ = map_name
        
        # 配置保持不变，导航内部根据地图名称和配置的目录自动解析掩码
        self.config_ = config
        self.config_.minimap.template_path = str(get_program_dir() / "resource" / "template" / GlobalContext().template_tier / "minimap_border.png")
        # 创建NavigationMain实例（延迟导入）
        NavigationMain = _get_navigation_main()
        self.nav_main_ = NavigationMain(self.config_, map_name=map_name)
        
        # 初始化
        if not self.nav_main_.Initialize():
            logger.error("NavigationMain 初始化失败")
            return False
        
        # 在独立线程中运行
        self.running_ = True
        self.thread_ = threading.Thread(target=self._run_thread, daemon=True)
        self.thread_.start()
        
        logger.info("AI控制器已启动")
        return True
    
    def stop(self) -> None:
        """停止AI控制器"""
        if not self.running_:
            return
        
        logger.info("正在停止AI控制器...")
        self.running_ = False
        
        if self.nav_main_:
            self.nav_main_.Stop()
        
        if self.thread_ and self.thread_.is_alive():
            self.thread_.join(timeout=2.0)
        
        self.nav_main_ = None
        logger.info("AI控制器已停止")
    
    def is_running(self) -> bool:
        """
        检查AI控制器是否正在运行
        
        Returns:
            是否正在运行
        """
        return self.running_ and self.nav_main_ is not None
    
    def _run_thread(self) -> None:
        """运行线程"""
        try:
            if self.nav_main_:
                self.nav_main_.Run()
        except Exception as e:
            logger.error(f"AI控制器运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running_ = False
    
