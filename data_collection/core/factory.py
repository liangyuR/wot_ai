"""
组件工厂 - 简化组件创建和配置
"""

from typing import Optional
from pathlib import Path
from loguru import logger

from data_collection.core.screen_capture import ScreenCapture
from data_collection.core.input_listener import InputListener


class ComponentFactory:
    """组件工厂，负责创建和配置各个组件"""
    
    @staticmethod
    def CreateScreenCapture(
        mode: str = "auto",
        width: int = 1920,
        height: int = 1080,
        prefer_dxcam: bool = True
    ) -> Optional[ScreenCapture]:
        """
        创建屏幕捕获器
        
        Args:
            mode: 'mss', 'dxcam', 'auto'（自动选择最佳）
            width: 屏幕宽度
            height: 屏幕高度
            prefer_dxcam: 是否优先使用 dxcam（如果可用）
            
        Returns:
            屏幕捕获器实例，失败返回 None
        """
        if mode == "auto":
            # 自动选择：优先 dxcam，否则 mss
            if prefer_dxcam:
                try:
                    from data_collection.capture.dxcam_capture import DxcamScreenCapture
                    capture = DxcamScreenCapture()
                    if capture.Initialize(width, height):
                        logger.info("✓ 使用 DXCam 屏幕捕获（高性能）")
                        return capture
                except Exception as e:
                    logger.debug(f"DXCam 不可用: {e}，降级到 MSS")
            
            # 降级到 MSS
            try:
                from data_collection.capture.mss_capture import MssScreenCapture
                capture = MssScreenCapture()
                if capture.Initialize(width, height):
                    logger.info("✓ 使用 MSS 屏幕捕获")
                    return capture
            except Exception as e:
                logger.error(f"MSS 初始化失败: {e}")
                return None
        
        elif mode == "dxcam":
            try:
                from data_collection.capture.dxcam_capture import DxcamScreenCapture
                capture = DxcamScreenCapture()
                if capture.Initialize(width, height):
                    return capture
            except Exception as e:
                logger.error(f"DXCam 初始化失败: {e}")
                return None
        
        elif mode == "mss":
            try:
                from data_collection.capture.mss_capture import MssScreenCapture
                capture = MssScreenCapture()
                if capture.Initialize(width, height):
                    return capture
            except Exception as e:
                logger.error(f"MSS 初始化失败: {e}")
                return None
        
        return None
    
    @staticmethod
    def CreateInputListener(use_global_hook: bool = True) -> Optional[InputListener]:
        """
        创建输入监听器
        
        Args:
            use_global_hook: 是否使用全局钩子（Windows Hook API）
            
        Returns:
            输入监听器实例，失败返回 None
        """
        if use_global_hook:
            try:
                from data_collection.listeners.global_listener import GlobalInputListener
                listener = GlobalInputListener()
                logger.info("✓ 使用全局输入监听器")
                return listener
            except Exception as e:
                logger.warning(f"全局钩子不可用: {e}，降级到 pynput")
        
        # 降级到 pynput
        try:
            from data_collection.listeners.pynput_listener import PynputInputListener
            listener = PynputInputListener()
            logger.info("✓ 使用 Pynput 输入监听器")
            return listener
        except Exception as e:
            logger.error(f"Pynput 初始化失败: {e}")
            return None
    
    @staticmethod
    def CreateFrameSaver(
        frames_dir: Path,
        prefer_png_saver: bool = True,
        png_compression: int = 3,
        queue_size: int = 60,
        jpeg_quality: int = None  # 已弃用，保留用于向后兼容
    ):
        """
        创建帧保存器（PNG格式）
        
        Args:
            frames_dir: 帧保存目录
            prefer_png_saver: 是否优先使用 PNG 保存器（默认 True，之前是 turbojpeg）
            png_compression: PNG 压缩级别 (0-9, 默认3)
            queue_size: 队列大小
            jpeg_quality: 已弃用，保留用于向后兼容
            
        Returns:
            帧保存器实例
        """
        # 为了向后兼容，如果提供了 jpeg_quality，转换为 PNG 压缩级别
        if jpeg_quality is not None:
            png_compression = max(0, min(9, 9 - int(jpeg_quality / 11)))
        
        if prefer_png_saver:
            try:
                from data_collection.save.turbojpeg_saver import PngFrameSaver
                saver = PngFrameSaver(png_compression=png_compression, queue_size=queue_size)
                if saver.Start():
                    logger.info("✓ 使用 PNG 帧保存器（高性能）")
                    return saver
            except Exception as e:
                logger.debug(f"PNG 保存器不可用: {e}，降级到 OpenCV")
        
        # 降级到 OpenCV PNG 保存器
        try:
            from data_collection.save.async_frame_saver import AsyncFrameSaver
            saver = AsyncFrameSaver(png_compression=png_compression, queue_size=queue_size)
            if saver.Start():
                logger.info("✓ 使用 OpenCV PNG 帧保存器")
                return saver
        except Exception as e:
            logger.error(f"帧保存器初始化失败: {e}")
            return None
        
        return None

