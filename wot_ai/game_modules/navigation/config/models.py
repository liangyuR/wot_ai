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
    path: str = Field(..., description="YOLO模型文件路径")
    conf_threshold: float = Field(..., description="置信度阈值")
    iou_threshold: float = Field(..., description="IoU阈值")
    
    @field_validator('path')
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
    template_path: str = Field(..., description="小地图模板文件路径")
    size: Tuple[int, int] = Field(..., description="小地图尺寸 (width, height)")
    max_size: int = Field(..., description="小地图最大尺寸")
    
    @field_validator('template_path')
    @classmethod
    def validate_template_path(cls, v: str) -> str:
        """验证模板路径是否存在"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"小地图模板文件不存在: {v}")
        return str(path.resolve())
    
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


class MaskConfig(BaseModel):
    """掩码配置"""
    path: Optional[str] = Field(None, description="掩码文件路径（可选）")
    directory: Optional[str] = Field(None, description="掩码文件夹（可选）")
    filename_format: str = Field(
        "{map_name}_mask.png",
        description="掩码文件命名模板，支持 {map_name} 占位符"
    )
    inflation_radius_px: int = Field(..., description="障碍膨胀半径（像素）")
    cost_alpha: float = Field(..., description="代价图权重")
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        """验证掩码路径（如果提供）"""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"掩码文件不存在: {v}")
            if not path.is_file():
                raise ValueError(f"掩码路径必须是文件: {v}")
            return str(path.resolve())
        return v
    
    @field_validator('directory')
    @classmethod
    def validate_directory(cls, v: Optional[str]) -> Optional[str]:
        """验证掩码文件夹（如果提供）"""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"掩码文件夹不存在: {v}")
            if not path.is_dir():
                raise ValueError(f"掩码文件夹必须是目录: {v}")
            return str(path.resolve())
        return v
    
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


class VehicleConfig(BaseModel):
    """车辆配置"""
    screenshot_dir: str = Field(..., description="车辆截图目录")
    
    @field_validator('screenshot_dir')
    @classmethod
    def validate_screenshot_dir(cls, v: str) -> str:
        """验证车辆截图目录"""
        path = Path(v)
        if path.exists() and not path.is_dir():
            raise ValueError(f"车辆截图路径必须是目录: {v}")
        return str(path.resolve() if path.exists() else path)


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
    move_speed: float = Field(..., description="移动速度")
    rotation_smooth: float = Field(..., description="旋转平滑度")
    target_point_offset: int = Field(..., description="目标点偏移量")
    path_deviation_tolerance: float = Field(..., description="路径偏离容忍度（像素）")
    goal_arrival_threshold: float = Field(..., description="终点到达阈值（像素）")
    stuck_threshold: float = Field(..., description="卡顿检测阈值（像素）")
    stuck_frames_threshold: int = Field(..., description="连续卡顿帧数阈值")
    
    @field_validator('move_speed', 'rotation_smooth')
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """验证正浮点数"""
        if v <= 0:
            raise ValueError(f"值必须大于0: {v}")
        return v
    
    @field_validator('target_point_offset', 'stuck_frames_threshold')
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """验证正整数"""
        if v <= 0:
            raise ValueError(f"值必须大于0: {v}")
        return v


class UIConfig(BaseModel):
    """UI配置"""
    overlay_fps: int = Field(..., description="覆盖层FPS")
    overlay_alpha: int = Field(..., description="覆盖层透明度 (0-255)")
    
    @field_validator('overlay_fps')
    @classmethod
    def validate_fps(cls, v: int) -> int:
        """验证FPS"""
        if v <= 0:
            raise ValueError(f"FPS必须大于0: {v}")
        return v
    
    @field_validator('overlay_alpha')
    @classmethod
    def validate_alpha(cls, v: int) -> int:
        """验证透明度"""
        if not 0 <= v <= 255:
            raise ValueError(f"透明度必须在0-255之间: {v}")
        return v


class NavigationConfig(BaseModel):
    """导航主配置"""
    model: ModelConfig = Field(..., description="YOLO模型配置")
    minimap: MinimapConfig = Field(..., description="小地图配置")
    mask: Optional[MaskConfig] = Field(None, description="掩码配置（可选）")
    vehicle: Optional[VehicleConfig] = Field(None, description="车辆配置（可选）")
    grid: GridConfig = Field(..., description="栅格配置")
    path_planning: PathPlanningConfig = Field(..., description="路径规划配置")
    control: ControlConfig = Field(..., description="控制配置")
    ui: UIConfig = Field(..., description="UI配置")
    monitor_index: int = Field(..., description="屏幕捕获监视器索引")
    
    @field_validator('monitor_index')
    @classmethod
    def validate_monitor_index(cls, v: int) -> int:
        """验证监视器索引"""
        if v < 0:
            raise ValueError(f"监视器索引不能为负数: {v}")
        return v

