#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测性能测试用例
实时检测self_pos并在UI中显示，用于测试检测性能
不包含路径规划和控制逻辑
"""

import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from loguru import logger

# 导入所需模块
from src.navigation.service.capture_service import CaptureService
from src.vision.detection.minimap_anchor_detector import MinimapAnchorDetector
from src.vision.detection.minimap_detector import MinimapDetector
from src.navigation.ui.transparent_overlay import TransparentOverlay


class DetectionPerformanceTest:
    """检测性能测试类"""
    
    def __init__(self, config: Dict):
        """
        初始化测试
        
        Args:
            config: 配置字典
        """
        self.config_ = config
        self.running_ = False
        
        # 服务对象
        self.capture_service_: Optional[CaptureService] = None
        self.anchor_detector_: Optional[MinimapAnchorDetector] = None
        self.minimap_detector_: Optional[MinimapDetector] = None
        self.overlay_: Optional[TransparentOverlay] = None
        
        # 小地图区域信息
        self.minimap_region_: Optional[Dict] = None
        
        # 使用队列传递数据（完全无锁）
        self.detection_queue_ = queue.Queue(maxsize=10)  # 支持30FPS
        
        # 线程对象
        self.detection_thread_: Optional[threading.Thread] = None
        self.ui_thread_: Optional[threading.Thread] = None
    
    def Initialize(self) -> bool:
        """
        初始化所有服务
        
        Returns:
            是否初始化成功
        """
        try:
            # 1. 初始化屏幕捕获服务
            logger.info("初始化屏幕捕获服务...")
            self.capture_service_ = CaptureService(monitor_index=1)
            
            # 2. 初始化小地图锚点检测器
            template_path = self.config_.get('minimap_template_path')
            if not template_path or not Path(template_path).exists():
                logger.error(f"小地图模板不存在: {template_path}")
                return False
            
            logger.info(f"初始化小地图锚点检测器: {template_path}")
            self.anchor_detector_ = MinimapAnchorDetector(
                template_path=template_path,
                debug=False,
                multi_scale=False
            )
            
            # 3. 初始化YOLO检测器
            model_path = self.config_.get('model_path')
            if not model_path or not Path(model_path).exists():
                logger.error(f"YOLO模型不存在: {model_path}")
                return False
            
            logger.info(f"初始化YOLO检测器: {model_path}")
            self.minimap_detector_ = MinimapDetector(
                model_path=model_path,
                conf_threshold=self.config_.get('conf_threshold', 0.25),
                iou_threshold=self.config_.get('iou_threshold', 0.75)
            )
            if not self.minimap_detector_.LoadModel():
                logger.error("YOLO模型加载失败")
                return False
            
            logger.info("所有服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def DetectMinimapRegion(self, frame: np.ndarray) -> Optional[Dict]:
        """
        检测小地图区域
        
        Args:
            frame: 屏幕帧
        
        Returns:
            小地图区域字典 {x, y, width, height}，失败返回 None
        """
        # 获取屏幕尺寸
        frame_h, frame_w = frame.shape[:2]
        
        # 使用默认尺寸进行模板匹配（用于检测位置）
        default_size = self.config_.get('minimap_size', 640)
        
        # 使用MinimapAnchorDetector检测小地图位置
        top_left = self.anchor_detector_.detect(frame, size=default_size)
        if top_left is None:
            logger.warning("无法检测到小地图位置")
            return None
        
        x, y = top_left
        
        # 自适应检测小地图尺寸
        minimap_size = self.config_.get('minimap_size', 640)
        
        region = {
            'x': x,
            'y': y,
            'width': minimap_size,
            'height': minimap_size
        }
        
        logger.info(f"检测到小地图区域: {region}")
        return region
    
    def InitializeOverlay(self, minimap_region: Dict) -> bool:
        """
        初始化透明覆盖层
        
        Args:
            minimap_region: 小地图区域 {x, y, width, height}
        
        Returns:
            是否初始化成功
        """
        try:
            overlay_width = minimap_region['width']
            overlay_height = minimap_region['height']
            logger.info(f"初始化透明覆盖层，位置: (0, 0)（左上角）, "
                       f"尺寸: {overlay_width}x{overlay_height}")
            
            self.overlay_ = TransparentOverlay(
                width=overlay_width,
                height=overlay_height,
                window_name="WOT_AI Detection Test",
                pos_x=0,
                pos_y=0,
                fps=self.config_.get('overlay_fps', 30),
                alpha=self.config_.get('overlay_alpha', 180)
            )
            
            logger.info("透明覆盖层初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"透明覆盖层初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ExtractMinimap(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从小地图区域提取小地图图像
        
        Args:
            frame: 屏幕帧
        
        Returns:
            小地图图像，失败返回 None
        """
        if self.minimap_region_ is None:
            return None
        
        x = self.minimap_region_['x']
        y = self.minimap_region_['y']
        w = self.minimap_region_['width']
        h = self.minimap_region_['height']
        
        # 提取小地图区域
        minimap = frame[y:y+h, x:x+w]
        return minimap
    
    def _DetectionThread_(self):
        """检测线程：负责屏幕捕获、小地图提取、YOLO检测"""
        logger.info("检测线程启动")
        
        detection_fps = 30  # 30FPS
        detection_interval = 1.0 / detection_fps  # 约33ms
        
        while self.running_:
            try:
                start_time = time.time()
                
                # 1. 捕获屏幕
                frame = self.capture_service_.Capture()
                if frame is None:
                    time.sleep(detection_interval)
                    continue
                
                # 2. 提取小地图
                minimap = self.ExtractMinimap(frame)
                if minimap is None:
                    time.sleep(detection_interval)
                    continue
                
                # 3. YOLO检测
                detections = self.minimap_detector_.Detect(minimap, debug=False)
                if detections.self_pos is None:
                    time.sleep(detection_interval)
                    continue
                
                # 在锁外执行copy操作
                minimap_copy = minimap.copy()
                
                # 非阻塞放入队列（队列满时丢弃旧数据）
                try:
                    self.detection_queue_.put_nowait((detections, minimap_copy))
                except queue.Full:
                    # 队列满，丢弃最旧的数据，放入新数据
                    try:
                        self.detection_queue_.get_nowait()
                        self.detection_queue_.put_nowait((detections, minimap_copy))
                    except queue.Empty:
                        pass
                
                # 控制循环频率
                elapsed = time.time() - start_time
                sleep_time = max(0, detection_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"检测线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(detection_interval)
        
        logger.info("检测线程退出")
    
    def _UIThread_(self):
        """UI更新线程：更新透明覆盖层显示（30FPS，完全无锁）"""
        logger.info("UI更新线程启动")
        
        ui_fps = 30  # 30FPS
        ui_interval = 1.0 / ui_fps  # 约33ms
        
        while self.running_:
            try:
                start_time = time.time()
                
                # 清空检测队列，只取最新的检测结果
                detections = None
                minimap = None
                while not self.detection_queue_.empty():
                    try:
                        detections, minimap = self.detection_queue_.get_nowait()
                    except queue.Empty:
                        break
                
                if detections is not None and minimap is not None and self.overlay_ is not None:
                    # 准备检测结果字典
                    detections_dict = {
                        'self_pos': detections.self_pos,
                        'flag_pos': detections.enemy_flag_pos,
                        'angle': detections.self_angle
                    }
                    
                    # 获取尺寸信息
                    minimap_h, minimap_w = minimap.shape[:2]
                    minimap_size = (minimap_w, minimap_h)
                    grid_size = self.config_.get('grid_size', (256, 256))
                    
                    # 更新覆盖层（路径传空列表，只显示检测结果）
                    self.overlay_.DrawPath(
                        minimap=minimap,
                        path=[],  # 空路径，不显示路径
                        detections=detections_dict,
                        minimap_size=minimap_size,
                        grid_size=grid_size,
                        mask=None  # 不显示掩码
                    )
                
                # 控制循环频率（30FPS）
                elapsed = time.time() - start_time
                sleep_time = max(0, ui_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"UI更新线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(ui_interval)
        
        logger.info("UI更新线程退出")
    
    def Run(self):
        """运行测试"""
        logger.info("开始检测性能测试...")
        
        # 1. 初始化服务
        if not self.Initialize():
            logger.error("初始化失败")
            return
        
        # 2. 检测小地图区域
        logger.info("正在检测小地图区域...")
        frame = self.capture_service_.Capture()
        if frame is None:
            logger.error("无法捕获屏幕")
            return
        
        self.minimap_region_ = self.DetectMinimapRegion(frame)
        if self.minimap_region_ is None:
            logger.error("无法检测到小地图区域")
            return
        
        # 3. 初始化Overlay
        if not self.InitializeOverlay(self.minimap_region_):
            logger.error("Overlay初始化失败")
            return
        
        # 4. 启动线程
        self.running_ = True
        self.detection_thread_ = threading.Thread(target=self._DetectionThread_, daemon=True)
        self.ui_thread_ = threading.Thread(target=self._UIThread_, daemon=True)
        
        self.detection_thread_.start()
        self.ui_thread_.start()
        
        logger.info("测试已启动，按Ctrl+C退出...")
        
        # 5. 主循环
        try:
            while self.running_:
                time.sleep(0.1)
                # 检查线程是否还在运行
                if not self.detection_thread_.is_alive():
                    logger.warning("检测线程已退出")
                if not self.ui_thread_.is_alive():
                    logger.warning("UI更新线程已退出")
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号")
        finally:
            logger.info("正在停止所有线程...")
            self.running_ = False
            
            # 等待线程结束
            if self.detection_thread_ and self.detection_thread_.is_alive():
                self.detection_thread_.join(timeout=1.0)
            if self.ui_thread_ and self.ui_thread_.is_alive():
                self.ui_thread_.join(timeout=1.0)
        
        # 清理资源
        if self.overlay_:
            self.overlay_.Close()
        
        logger.info("测试结束")
    
    def Stop(self):
        """停止运行"""
        logger.info("正在停止...")
        self.running_ = False
        
        # 等待线程结束
        if self.detection_thread_ and self.detection_thread_.is_alive():
            self.detection_thread_.join(timeout=1.0)
        if self.ui_thread_ and self.ui_thread_.is_alive():
            self.ui_thread_.join(timeout=1.0)
        
        if self.overlay_:
            self.overlay_.Close()
        
        logger.info("已停止")


def main():
    """主函数"""
    script_dir = Path(__file__).resolve().parent
    
    # 配置参数
    config = {
        'model_path': str(script_dir / "best.pt"),
        'minimap_template_path': str(script_dir / "minimap_border.png"),
        'conf_threshold': 0.25,
        'iou_threshold': 0.75,
        'minimap_size': 640,
        'grid_size': (256, 256),
        'overlay_fps': 30,
        'overlay_alpha': 180
    }
    
    # 检查文件是否存在
    if not Path(config['model_path']).exists():
        logger.error(f"模型文件不存在: {config['model_path']}")
        return
    
    if not Path(config['minimap_template_path']).exists():
        logger.error(f"小地图模板不存在: {config['minimap_template_path']}")
        return
    
    # 创建测试实例并运行
    test = DetectionPerformanceTest(config)
    test.Run()


if __name__ == "__main__":
    main()

