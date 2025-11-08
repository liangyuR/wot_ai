"""
内存池 - 用于对象复用，减少内存分配
"""

from typing import List, Optional
import numpy as np
from loguru import logger


class FramePool:
    """帧对象池 - 复用 numpy 数组以减少内存分配"""
    
    def __init__(self, pool_size: int = 5, frame_shape: tuple = None):
        """
        初始化帧池
        
        Args:
            pool_size: 池大小
            frame_shape: 帧形状 (height, width, channels)，如果为 None 则动态分配
        """
        self.pool_size_ = pool_size
        self.frame_shape_ = frame_shape
        self.pool_ = []
        self.in_use_ = set()
    
    def Get(self, shape: tuple = None) -> Optional[np.ndarray]:
        """
        从池中获取一个帧缓冲区
        
        Args:
            shape: 帧形状，如果为 None 使用默认形状
            
        Returns:
            numpy 数组，如果池为空则创建新的
        """
        target_shape = shape or self.frame_shape_
        
        # 尝试从池中获取
        for i, frame in enumerate(self.pool_):
            if frame.shape == target_shape and id(frame) not in self.in_use_:
                self.pool_.pop(i)
                self.in_use_.add(id(frame))
                return frame
        
        # 池为空，创建新的
        if target_shape:
            frame = np.empty(target_shape, dtype=np.uint8)
            self.in_use_.add(id(frame))
            return frame
        
        return None
    
    def Return(self, frame: np.ndarray):
        """
        归还帧到池中
        
        Args:
            frame: 要归还的帧
        """
        frame_id = id(frame)
        if frame_id in self.in_use_:
            self.in_use_.discard(frame_id)
            
            # 如果池未满，归还
            if len(self.pool_) < self.pool_size_:
                self.pool_.append(frame)
    
    def Clear(self):
        """清空池"""
        self.pool_.clear()
        self.in_use_.clear()


class EventPool:
    """事件对象池 - 复用事件对象以减少分配"""
    
    def __init__(self, pool_size: int = 20):
        """
        初始化事件池
        
        Args:
            pool_size: 池大小
        """
        self.pool_size_ = pool_size
        self.pool_ = []
        self.in_use_ = set()
    
    def Get(self, event_type, data: dict = None):
        """
        从池中获取事件对象
        
        Args:
            event_type: 事件类型
            data: 事件数据
            
        Returns:
            事件对象
        """
        # 简单实现：直接创建新对象
        # 高级实现可以使用池化，但需要实现事件对象的重置方法
        try:
            from data_collection.recording_events import RecordingEvent
            event = RecordingEvent(event_type, data or {})
            return event
        except ImportError:
            return None
    
    def Return(self, event):
        """归还事件对象（占位，未来实现）"""
        pass
    
    def Clear(self):
        """清空池"""
        self.pool_.clear()
        self.in_use_.clear()

