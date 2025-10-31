"""
PNG 帧保存器（使用 OpenCV，替代 TurboJPEG）
注意：TurboJPEG 仅支持 JPEG，已废弃。PNG 格式提供无损质量。
"""

import cv2
import queue
import threading
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger

from data_collection.save.frame_saver import FrameSaver


class PngFrameSaver(FrameSaver):
    """使用 OpenCV 的异步 PNG 帧保存器（替代 TurboJPEG，支持 PNG 格式）"""
    
    def __init__(self, png_compression: int = 3, queue_size: int = 60, jpeg_quality: int = None):
        """
        初始化 PNG 保存器
        
        Args:
            png_compression: PNG 压缩级别 (0-9, 0=无压缩最快, 9=最高压缩最慢, 默认3)
            queue_size: 队列大小（帧数）
            jpeg_quality: 已弃用，保留用于向后兼容，实际使用 png_compression
        """
        # 为了向后兼容，如果提供了 jpeg_quality，转换为 PNG 压缩级别
        if jpeg_quality is not None:
            # JPEG 质量 0-100 映射到 PNG 压缩级别 0-9 (反向映射，高质量=低压缩)
            self.png_compression_ = max(0, min(9, 9 - int(jpeg_quality / 11)))
            logger.warning("jpeg_quality 参数已弃用，请使用 png_compression (0-9)")
        else:
            self.png_compression_ = png_compression
        
        self.queue_ = queue.Queue(maxsize=queue_size)
        self.running_ = False
        self.thread_ = None
        
        # 统计
        self.total_saved_ = 0
        self.total_dropped_ = 0
        
        logger.info(f"PNG 帧保存器初始化成功（压缩级别: {self.png_compression_}, 队列: {queue_size}）")
    
    def Start(self):
        """启动保存线程"""
        if self.running_:
            return
        
        self.running_ = True
        self.thread_ = threading.Thread(target=self._saveWorker, daemon=True)
        self.thread_.start()
        logger.info(f"PNG 异步保存器已启动（队列大小: {self.queue_.maxsize}）")
        return True
    
    def Stop(self):
        """停止保存线程并等待队列清空"""
        if not self.running_:
            return
        
        logger.info(f"停止 PNG 保存器，队列中还有 {self.queue_.qsize()} 帧...")
        
        # 等待队列清空
        try:
            self.queue_.join(timeout=10.0)
            logger.info("队列已清空")
        except Exception as e:
            logger.warning(f"等待队列清空时出错: {e}")
        
        # 停止线程
        self.running_ = False
        self.queue_.put(None)  # 发送停止信号
        
        if self.thread_:
            self.thread_.join(timeout=5.0)
        
        logger.info(f"PNG 保存器已停止（已保存: {self.total_saved_}, 丢弃: {self.total_dropped_}）")
    
    def SaveFrame(self, frame: np.ndarray, frame_number: int, output_path: Path) -> bool:
        """
        添加帧到保存队列
        
        Args:
            frame: BGR 格式的帧
            frame_number: 帧号
            output_path: 输出路径（会自动创建目录，必须是 .png 扩展名）
            
        Returns:
            是否成功添加到队列
        """
        try:
            # 确保输出路径是 PNG 格式
            if output_path.suffix.lower() != '.png':
                output_path = output_path.with_suffix('.png')
            
            # 使用 put_nowait 避免阻塞
            # 需要复制帧，因为 frame 可能被重用
            frame_copy = frame.copy()  # 线程安全需要的复制
            self.queue_.put_nowait((frame_copy, frame_number, output_path))
            return True
        except queue.Full:
            self.total_dropped_ += 1
            if self.total_dropped_ % 10 == 1:
                logger.warning(f"保存队列已满，丢弃帧 {frame_number}（累计丢弃: {self.total_dropped_}）")
            return False
    
    def _saveWorker(self):
        """保存线程工作函数"""
        while self.running_:
            try:
                item = self.queue_.get(timeout=1.0)
                
                if item is None:  # 停止信号
                    self.queue_.task_done()
                    break
                
                frame, frame_number, output_path = item
                
                # 确保目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 使用 OpenCV 保存为 PNG 格式
                # PNG 压缩级别 0-9，0=无压缩最快，9=最高压缩最慢
                try:
                    success = cv2.imwrite(
                        str(output_path),
                        frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression_]
                    )
                    
                    if success:
                        self.total_saved_ += 1
                    else:
                        logger.error(f"保存帧 {frame_number} 失败")
                except Exception as e:
                    logger.error(f"保存帧 {frame_number} 失败: {e}")
                
                self.queue_.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"保存帧时出错: {e}")
                if item:
                    self.queue_.task_done()
    
    def GetStats(self) -> dict:
        """获取统计信息"""
        return {
            "saved": self.total_saved_,
            "dropped": self.total_dropped_,
            "queue_size": self.queue_.qsize(),
            "queue_max": self.queue_.maxsize
        }
    
    def GetQueueSize(self) -> int:
        """获取当前队列大小"""
        return self.queue_.qsize()

