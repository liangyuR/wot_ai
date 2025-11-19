#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI控制器模块

封装 NavigationMain 为战斗中的AI控制器，根据地图名称自动加载对应的mask掩码。
"""

import threading
from typing import Optional
from loguru import logger
from src.navigation.config.models import NavigationConfig
from src.navigation.navigation_main import NavigationMain
from src.utils.global_path import MinimapBorderTemplatePath

# 延迟导入NavigationMain以避免循环导入
def _get_navigation_main():
    from src.navigation.navigation_main import NavigationMain
    return NavigationMain


class AIController:
    """AI控制器"""
    
    def __init__(self):
        """初始化AI控制器"""
        self.nav_main_: Optional['NavigationMain'] = None
        self.config_: Optional[NavigationConfig] = None
        self.map_name_: Optional[str] = None
        self.initialized_ = False  # 是否已初始化（YOLO模型已加载）
        self.running_ = False  # 是否正在运行（线程是否活跃）
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
        
        # 如果未初始化，执行完整初始化
        if not self.initialized_:
            # 配置保持不变，导航内部根据地图名称和配置的目录自动解析掩码
            self.config_ = config
            self.config_.minimap.template_path = MinimapBorderTemplatePath()
            # 创建NavigationMain实例（延迟导入）
            NavigationMain = _get_navigation_main()
            self.nav_main_ = NavigationMain(self.config_, map_name=map_name)
            
            # 初始化
            if not self.nav_main_.Initialize():
                logger.error("NavigationMain 初始化失败")
                return False
            
            self.initialized_ = True
            logger.info("导航AI初始化完成（YOLO模型已加载）")
        else:
            # 已初始化，只更新地图名称
            if self.nav_main_:
                self.nav_main_.set_map_name(map_name)
            logger.info("导航AI已初始化，跳过初始化步骤")
        
        # 在独立线程中运行
        self.running_ = True
        self.thread_ = threading.Thread(target=self._run_thread, daemon=True)
        self.thread_.start()
        
        logger.info("AI控制器已启动")
        return True
    
    def stop(self) -> None:
        """
        停止AI控制器运行循环（保留初始化状态）
        """
        if not self.running_:
            return
        
        logger.info("正在停止AI控制器运行循环...")
        self.running_ = False
        
        if self.nav_main_:
            self.nav_main_.Stop()
        
        if self.thread_ and self.thread_.is_alive():
            self.thread_.join(timeout=2.0)
        
        # 不销毁 nav_main_，保留初始化状态
        logger.info("AI控制器运行循环已停止（保留初始化状态）")
    
    def update_mask(self, map_name: str) -> bool:
        """
        更新地图名称并重新加载掩码
        
        Args:
            map_name: 新的地图名称
        
        Returns:
            是否成功更新
        """
        if not self.initialized_ or not self.nav_main_:
            logger.warning("导航AI未初始化，无法更新掩码")
            return False
        
        if self.running_:
            logger.warning("导航AI正在运行，无法更新掩码，请先停止")
            return False
        
        logger.info(f"更新掩码: {self.map_name_} -> {map_name}")
        self.map_name_ = map_name
        
        # 调用 NavigationMain 的 update_mask 方法
        if not self.nav_main_.update_mask(map_name):
            logger.error("掩码更新失败")
            return False
        
        logger.info("掩码更新成功")
        return True
    
    def is_running(self) -> bool:
        """
        检查AI控制器是否正在运行
        
        Returns:
            是否正在运行
        """
        return self.running_ and self.nav_main_ is not None
    
    def is_initialized(self) -> bool:
        """
        检查AI控制器是否已初始化
        
        Returns:
            是否已初始化
        """
        return self.initialized_ and self.nav_main_ is not None
    
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
    
