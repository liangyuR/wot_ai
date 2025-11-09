#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
常量定义：集中管理所有魔法数字和配置常量
"""

# =============================
# 检测相关常量
# =============================

# 默认置信度阈值
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25

# 性能模式对应的置信度阈值
CONFIDENCE_HIGH: float = 0.25
CONFIDENCE_MEDIUM: float = 0.35
CONFIDENCE_LOW: float = 0.45

# 位置平滑器窗口大小
SMOOTHER_WINDOW_SELF_POS: int = 3
SMOOTHER_WINDOW_FLAG_POS: int = 3
SMOOTHER_WINDOW_ANGLE: int = 5

# 静态掩码膨胀像素数
DEFAULT_INFLATE_PX: int = 4

# =============================
# 路径规划相关常量
# =============================

# 默认栅格尺寸
DEFAULT_GRID_SIZE: tuple[int, int] = (64, 64)

# 路径平滑权重
SMOOTH_WEIGHT_DEFAULT: float = 0.3
SMOOTH_WEIGHT_LOW: float = 0.2
SMOOTH_WEIGHT_MEDIUM: float = 0.3
SMOOTH_WEIGHT_HIGH: float = 0.35

# A*算法移动方向（四方向）
DIRECTIONS_4WAY = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# =============================
# UI和显示相关常量
# =============================

# 默认小地图尺寸
DEFAULT_MINIMAP_WIDTH: int = 320
DEFAULT_MINIMAP_HEIGHT: int = 320

# 透明浮窗默认透明度
DEFAULT_OVERLAY_ALPHA: int = 180

# =============================
# 性能配置常量
# =============================

# 性能模式对应的FPS
FPS_HIGH: int = 30
FPS_MEDIUM: int = 15
FPS_LOW: int = 8

# 性能模式对应的缩放因子
SCALE_HIGH: float = 1.0
SCALE_MEDIUM: float = 0.75
SCALE_LOW: float = 0.6

# =============================
# 控制相关常量
# =============================

# 默认鼠标灵敏度
DEFAULT_MOUSE_SENSITIVITY: float = 1.0

# 默认标定系数
DEFAULT_CALIBRATION_FACTOR: float = 150.0

# 默认移动速度
DEFAULT_MOVE_SPEED: float = 1.0

# 默认旋转平滑系数
DEFAULT_ROTATION_SMOOTH: float = 0.3

# =============================
# 地图相关常量
# =============================

# 默认地图目录
DEFAULT_MAPS_DIR: str = "data/maps"

# 默认栅格缩放因子
DEFAULT_SCALE_FACTOR: float = 1.0

# =============================
# 其他常量
# =============================

# 线程超时时间（秒）
THREAD_JOIN_TIMEOUT: float = 2.0

# 默认更新间隔（秒）
DEFAULT_UPDATE_INTERVAL: float = 0.1

# 最大前进持续时间（秒）
MAX_MOVE_DURATION: float = 2.0

# 默认前进持续时间（秒）
DEFAULT_MOVE_DURATION: float = 0.5

