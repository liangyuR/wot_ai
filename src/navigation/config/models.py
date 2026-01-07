#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航配置模型

使用Pydantic定义类型安全的配置模型，所有字段都是必需的（无默认值）。
"""

from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator, AliasChoices
from pathlib import Path


class ModelConfig(BaseModel):
    """YOLO模型配置"""
    base_path: str = Field(..., description="基地模型文件路径")
    arrow_path: str = Field(..., description="箭头模型文件路径")
    conf_threshold: float = Field(..., description="置信度阈值")
    iou_threshold: float = Field(..., description="IoU阈值")

    class_id_flag: int = Field(..., description="旗帜类别ID")
    class_id_arrow: int = Field(..., description="箭头类别ID")
    
    @field_validator('base_path', 'arrow_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """验证模型路径是否存在"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"模型文件不存在: {v}")
        return str(path.resolve())
    
    @field_validator('conf_threshold')
    @classmethod
    def validate_conf_threshold(cls, v: float) -> float:
        """验证置信度阈值范围"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"置信度阈值必须在0.0-1.0之间: {v}")
        return v
    
    @field_validator('iou_threshold')
    @classmethod
    def validate_iou_threshold(cls, v: float) -> float:
        """验证IoU阈值范围"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"IoU阈值必须在0.0-1.0之间: {v}")
        return v


class MinimapConfig(BaseModel):
    """小地图配置"""
    size: Tuple[int, int] = Field(..., description="小地图尺寸 (width, height)")
    max_size: int = Field(..., description="小地图最大尺寸")
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """验证尺寸"""
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError(f"小地图尺寸必须大于0: {v}")
        return v
    
    @field_validator('max_size')
    @classmethod
    def validate_max_size(cls, v: int) -> int:
        """验证最大尺寸"""
        if v <= 0:
            raise ValueError(f"最大尺寸必须大于0: {v}")
        return v


class AngleDetectionConfig(BaseModel):
    """角度检测配置"""
    smoothing_alpha: float = Field(0.25, description="角度平滑系数 (0.0-1.0)")
    max_step_deg: float = Field(45.0, description="单帧最大角度变化（度）")
    # 自适应 alpha 阈值配置
    noise_threshold_deg: float = Field(2.0, description="噪声阈值（度），小于此值视为噪声")
    normal_threshold_deg: float = Field(10.0, description="正常转向阈值（度），小于此值视为正常转向")
    noise_alpha_factor: float = Field(0.4, description="噪声时的 alpha 缩放因子")
    large_turn_alpha_factor: float = Field(2.0, description="大幅转向时的 alpha 缩放因子")
    min_area_ratio: float = Field(0.2, description="轮廓面积最小比例")
    max_area_ratio: float = Field(0.9, description="轮廓面积最大比例")
    min_aspect_ratio: float = Field(0.3, description="外接矩形最小宽高比")
    max_aspect_ratio: float = Field(3.0, description="外接矩形最大宽高比")
    
    @field_validator('smoothing_alpha')
    @classmethod
    def validate_smoothing_alpha(cls, v: float) -> float:
        """验证平滑系数范围"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"平滑系数必须在0.0-1.0之间: {v}")
        return v
    
    @field_validator('max_step_deg', 'noise_threshold_deg', 'normal_threshold_deg', 'min_area_ratio', 'max_area_ratio', 'min_aspect_ratio', 'max_aspect_ratio')
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """验证正浮点数"""
        if v <= 0:
            raise ValueError(f"值必须大于0: {v}")
        return v
    
    @field_validator('noise_alpha_factor', 'large_turn_alpha_factor')
    @classmethod
    def validate_alpha_factor(cls, v: float) -> float:
        """验证 alpha 缩放因子"""
        if v <= 0:
            raise ValueError(f"alpha 缩放因子必须大于0: {v}")
        return v
    
    @field_validator('min_area_ratio', 'max_area_ratio')
    @classmethod
    def validate_area_ratio_range(cls, v: float) -> float:
        """验证面积比例范围"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"面积比例必须在0.0-1.0之间: {v}")
        return v


class MaskConfig(BaseModel):
    """掩码配置"""
    filename_format: str = Field(
        "{map_name}_mask.png",
        description="掩码文件命名模板，支持 {map_name} 占位符"
    )
    inflation_radius_px: int = Field(..., description="障碍膨胀半径（像素）")
    cost_alpha: float = Field(..., description="代价图权重")
    soft_obstacle_cost: float = Field(
        500.0,
        description="软障碍（膨胀区域）的代价，允许紧急穿越但代价极高"
    )
    
    @field_validator('inflation_radius_px')
    @classmethod
    def validate_inflation_radius(cls, v: int) -> int:
        """验证膨胀半径"""
        if v < 0:
            raise ValueError(f"膨胀半径不能为负数: {v}")
        return v
    
    @field_validator('cost_alpha')
    @classmethod
    def validate_cost_alpha(cls, v: float) -> float:
        """验证代价图权重"""
        if v < 0:
            raise ValueError(f"代价图权重不能为负数: {v}")
        return v
    
    @field_validator('soft_obstacle_cost')
    @classmethod
    def validate_soft_obstacle_cost(cls, v: float) -> float:
        """验证软障碍代价"""
        if v <= 0:
            raise ValueError(f"软障碍代价必须为正数: {v}")
        return v


class GridConfig(BaseModel):
    """栅格配置"""
    size: Tuple[int, int] = Field(..., description="栅格尺寸 (width, height)")
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """验证栅格尺寸"""
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError(f"栅格尺寸必须大于0: {v}")
        return v


class PathPlanningConfig(BaseModel):
    """路径规划配置"""
    enable_astar_smoothing: bool = Field(
        ...,
        description="是否启用A*内部平滑",
        validation_alias=AliasChoices("enable_astar_smoothing", "enable_smoothing")
    )
    astar_smooth_weight: float = Field(
        ...,
        description="A*平滑权重",
        validation_alias=AliasChoices("astar_smooth_weight", "smooth_weight")
    )
    post_smoothing_method: str = Field(
        ...,
        description="路径后处理方法: 'catmull_rom' 或 'los'",
        validation_alias=AliasChoices("post_smoothing_method", "smoothing_method")
    )
    simplify_method: str = Field(..., description="简化方法: 'rdp' 或其他")
    simplify_threshold: float = Field(..., description="简化阈值")
    num_points_per_segment: int = Field(..., description="每段采样点数")
    curvature_threshold_deg: float = Field(..., description="曲率阈值（度）")
    check_curvature: bool = Field(..., description="是否进行曲率检查")
    enable_los_post_smoothing: bool = Field(
        ...,
        description="当后处理方法为LOS时是否启用",
        validation_alias=AliasChoices("enable_los_post_smoothing", "enable_los_smoothing")
    )
    
    @field_validator('post_smoothing_method')
    @classmethod
    def validate_post_smoothing_method(cls, v: str) -> str:
        """验证平滑方法"""
        if v not in ['catmull_rom', 'los']:
            raise ValueError(f"平滑方法必须是 'catmull_rom' 或 'los': {v}")
        return v
    
    @field_validator('astar_smooth_weight')
    @classmethod
    def validate_astar_smooth_weight(cls, v: float) -> float:
        """验证平滑权重"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"平滑权重必须在0.0-1.0之间: {v}")
        return v


class ControlConfig(BaseModel):
    """控制配置"""
    path_deviation_tolerance: float = Field(..., description="路径偏离容忍度（像素）")
    goal_arrival_threshold: float = Field(..., description="终点到达阈值（像素）")
    stuck_threshold: float = Field(..., description="卡顿检测阈值（像素）")

    # MovementController 参数
    angle_dead_zone_deg: float = Field(3.0, description="角度死区（度）")
    angle_slow_turn_deg: float = Field(15.0, description="角度慢转阈值（度）")
    distance_stop_threshold: float = Field(5.0, description="距离停止阈值（像素）")
    slow_down_distance: float = Field(30.0, description="开始减速距离（像素）")
    max_forward_speed: float = Field(1.0, description="最大前进速度")
    min_forward_factor: float = Field(0.3, description="最小前进因子")
    large_angle_threshold_deg: float = Field(60.0, description="大角度阈值（度）")
    large_angle_speed_reduction: float = Field(0.5, description="大角度速度衰减系数")
    corridor_ref_width: float = Field(40.0, description="横向误差归一化宽度（像素）")
    k_lat_normal: float = Field(0.3, description="走廊内横向增益")
    k_lat_edge: float = Field(0.5, description="走廊边缘横向增益")
    k_lat_recenter: float = Field(0.8, description="纠偏模式横向增益")
    straight_angle_enter_deg: float = Field(6.0, description="直行模式进入角度阈值（度）")
    straight_angle_exit_deg: float = Field(10.0, description="直行模式退出角度阈值（度）")
    straight_lat_enter: float = Field(25.0, description="直行模式进入横向阈值（像素）")
    straight_lat_exit: float = Field(35.0, description="直行模式退出横向阈值（像素）")
    edge_speed_reduction: float = Field(0.85, description="走廊边缘速度缩放")
    recenter_speed_reduction: float = Field(0.6, description="纠偏模式速度缩放")
    debug_log_interval: int = Field(30, description="控制日志输出间隔（帧）")
    
    # MoveExecutor 参数
    smoothing_alpha: float = Field(0.3, description="平滑滤波系数（已废弃，保留兼容）")
    smoothing_alpha_forward: float = Field(0.2, description="前进平滑系数")
    smoothing_alpha_turn: float = Field(0.6, description="转向平滑系数")
    turn_deadzone: float = Field(0.12, description="转向死区")
    min_hold_time_ms: float = Field(100.0, description="最小按键保持时间（毫秒）")
    forward_hysteresis_on: float = Field(0.35, description="前进滞回开启阈值")
    forward_hysteresis_off: float = Field(0.08, description="前进滞回关闭阈值")
    
    # PathFollower 参数
    max_lateral_error: float = Field(80.0, description="最大横向误差（像素）")
    lookahead_distance: float = Field(60.0, description="前瞻距离（像素）")
    waypoint_switch_radius: float = Field(20.0, description="Waypoint切换半径（像素）")

    @field_validator('corridor_ref_width', 'straight_lat_enter', 'straight_lat_exit')
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """验证正浮点数"""
        if v <= 0:
            raise ValueError(f"值必须大于0: {v}")
        return v
    
    @field_validator('smoothing_alpha', 'smoothing_alpha_forward', 'smoothing_alpha_turn', 'min_forward_factor', 'large_angle_speed_reduction', 'edge_speed_reduction', 'recenter_speed_reduction')
    @classmethod
    def validate_range_0_1(cls, v: float) -> float:
        """验证0-1范围"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"值必须在0.0-1.0之间: {v}")
        return v
    
    @field_validator('debug_log_interval')
    @classmethod
    def validate_debug_interval(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"debug_log_interval 必须大于0: {v}")
        return v


class UiConfig(BaseModel):
    """UI配置"""
    enable: bool = Field(False, description="是否启用调试视图")


class GameConfig(BaseModel):
    """游戏配置"""
    process_name: str = Field("WorldOfTanks.exe", description="游戏进程名称")
    exe_path: str = Field("C:/Games/World_of_Tanks_CN/WorldOfTanks.exe", description="游戏可执行文件路径")
    restart_wait_seconds: int = Field(10, description="重启等待时间（秒）")
    stuck_timeout_seconds: int = Field(480, description="卡死判定超时时间（秒）")
    enable_silver_reserve: bool = Field(False, description="是否启用银币储备")


class StuckDetectionConfig(BaseModel):
    """卡顿脱困配置"""
    reverse_duration_s: float = Field(0.8, description="脱困倒退时间（秒）")
    max_stuck_count: int = Field(3, description="连续卡顿帧数阈值")
    check_interval_s: float = Field(5.0, description="检测间隔（秒）")
    dist_threshold_px: float = Field(10.0, description="最小移动距离（像素）")


class PeriodicReplanConfig(BaseModel):
    """定期重规划配置"""
    enable: bool = Field(False, description="是否启用定期重规划")
    interval_s: float = Field(3.0, description="重规划间隔（秒）")
    
    @field_validator('interval_s')
    @classmethod
    def validate_interval_s(cls, v: float) -> float:
        """验证重规划间隔"""
        if v <= 0:
            raise ValueError(f"重规划间隔必须大于0: {v}")
        return v


class DetectionConfig(BaseModel):
    """检测配置"""
    max_fps: float = Field(60.0, description="检测帧率上限")

    @field_validator('max_fps')
    @classmethod
    def validate_max_fps(cls, v: float) -> float:
        """验证检测帧率"""
        if v <= 0:
            raise ValueError(f"检测帧率必须大于0: {v}")
        return v


class ScheduledRestartConfig(BaseModel):
    """定时重启配置"""
    enable: bool = Field(False, description="是否启用定时重启")
    interval_hours: float = Field(1.0, description="重启间隔（小时）")

    @field_validator('interval_hours')
    @classmethod
    def validate_interval_hours(cls, v: float) -> float:
        """验证重启间隔"""
        if v <= 0:
            raise ValueError(f"重启间隔必须大于0: {v}")
        return v


class AutoStopConfig(BaseModel):
    """自动停止配置"""
    enable: bool = Field(False, description="是否启用自动停止")
    run_hours: float = Field(24.0, description="运行时长限制（小时）")
    auto_shutdown: bool = Field(False, description="达到时长后是否自动关机")

    @field_validator('run_hours')
    @classmethod
    def validate_run_hours(cls, v: float) -> float:
        """验证运行时长"""
        if v <= 0:
            raise ValueError(f"运行时长必须大于0: {v}")
        return v


class NavigationConfig(BaseModel):
    """导航主配置"""
    model: ModelConfig = Field(..., description="YOLO模型配置")
    game: GameConfig = Field(default_factory=GameConfig, description="游戏配置")
    minimap: MinimapConfig = Field(..., description="小地图配置")
    mask: Optional[MaskConfig] = Field(None, description="掩码配置（可选）")
    grid: GridConfig = Field(..., description="栅格配置")
    path_planning: PathPlanningConfig = Field(..., description="路径规划配置")
    control: ControlConfig = Field(..., description="控制配置")
    angle_detection: Optional[AngleDetectionConfig] = Field(None, description="角度检测配置（可选）")
    monitor_index: int = Field(..., description="屏幕捕获监视器索引")
    ui: UiConfig = Field(..., description="UI配置")
    stuck_detection: StuckDetectionConfig = Field(..., description="卡顿脱困配置")
    periodic_replan: PeriodicReplanConfig = Field(default_factory=PeriodicReplanConfig, description="定期重规划配置")
    detection: DetectionConfig = Field(default_factory=DetectionConfig, description="检测配置")
    scheduled_restart: ScheduledRestartConfig = Field(default_factory=ScheduledRestartConfig, description="定时重启配置")
    auto_stop: AutoStopConfig = Field(default_factory=AutoStopConfig, description="自动停止配置")
    
    @field_validator('monitor_index')
    @classmethod
    def validate_monitor_index(cls, v: int) -> int:
        """验证监视器索引"""
        if v < 0:
            raise ValueError(f"监视器索引不能为负数: {v}")
        return v

