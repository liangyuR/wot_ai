#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角度平滑器：平滑角度变化，避免抖动

特性：
- 处理角度跨越 0°/360° 边界的情况
- 自适应平滑系数：根据角度变化幅度动态调整
- 输入验证：检查 NaN、inf 和角度范围
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


class AngleSmoother:
    """角度平滑器：平滑角度变化，避免抖动"""

    def __init__(
        self,
        alpha: float = 0.25,
        max_step_deg: float = 45.0,
        noise_threshold_deg: float = 2.0,
        normal_threshold_deg: float = 10.0,
        noise_alpha_factor: float = 0.4,
        large_turn_alpha_factor: float = 2.0,
    ):
        """初始化角度平滑器

        Args:
            alpha: 平滑系数，0 < alpha <= 1，值越小越平滑但响应越慢
            max_step_deg: 单帧最大允许变化角度（度），超过此值会被限幅
            noise_threshold_deg: 噪声阈值（度），小于此值视为噪声
            normal_threshold_deg: 正常转向阈值（度），小于此值视为正常转向
            noise_alpha_factor: 噪声时的 alpha 缩放因子
            large_turn_alpha_factor: 大幅转向时的 alpha 缩放因子
        """
        self.alpha_ = max(0.0, min(1.0, alpha))
        self.max_step_deg_ = max(0.0, max_step_deg)
        self.noise_threshold_deg_ = max(0.0, noise_threshold_deg)
        self.normal_threshold_deg_ = max(self.noise_threshold_deg_, normal_threshold_deg)
        self.noise_alpha_factor_ = max(0.0, noise_alpha_factor)
        self.large_turn_alpha_factor_ = max(1.0, large_turn_alpha_factor)

        self.angle_: Optional[float] = None
        self.valid_: bool = False

    def _normalizeAngle(self, angle: float) -> float:
        """将角度归一化到 [0, 360) 范围"""
        return angle % 360.0

    def _computeAngleDiff(self, new_angle: float, old_angle: float) -> float:
        """计算角度差，处理跨越 0°/360° 边界的情况

        Returns:
            角度差（度），范围 [-180, 180]
        """
        return (new_angle - old_angle + 540.0) % 360.0 - 180.0

    def _validateAngle(self, angle: Optional[float]) -> bool:
        """验证角度值是否有效

        Args:
            angle: 待验证的角度值

        Returns:
            True 如果角度有效，False 否则
        """
        if angle is None:
            return False
        if not np.isfinite(angle):
            return False
        # 角度应该在合理范围内（允许负值，会在归一化时处理）
        return abs(angle) < 1e6

    def _computeAdaptiveAlpha(self, abs_diff: float) -> float:
        """根据角度变化幅度计算自适应 alpha

        Args:
            abs_diff: 角度变化的绝对值（度）

        Returns:
            有效的 alpha 值 [0, 1]
        """
        if abs_diff < self.noise_threshold_deg_:
            # 极小变化，视为噪声，减弱更新
            effective_alpha = self.alpha_ * self.noise_alpha_factor_
        elif abs_diff < self.normal_threshold_deg_:
            # 正常缓慢转向，使用原始 alpha
            effective_alpha = self.alpha_
        else:
            # 大幅转向，放大 alpha，加快跟踪
            effective_alpha = min(1.0, self.alpha_ * self.large_turn_alpha_factor_)

        return max(0.0, min(1.0, effective_alpha))

    def Update(self, raw_angle: Optional[float]) -> Optional[float]:
        """更新角度，返回平滑后的角度

        Args:
            raw_angle: 原始角度（度），如果为 None 则返回当前有效角度

        Returns:
            平滑后的角度（度）或 None（如果从未有有效输入）
        """
        # 如果输入为 None，返回当前有效角度（保持连续性）
        if raw_angle is None:
            return self.angle_ if self.valid_ else None

        # 验证输入
        if not self._validateAngle(raw_angle):
            logger.debug(f"AngleSmoother: 无效的角度输入 {raw_angle}，使用当前角度")
            return self.angle_ if self.valid_ else None

        # 归一化角度到 [0, 360)
        normalized_angle = self._normalizeAngle(raw_angle)

        # 首次有效输入，直接使用
        if not self.valid_:
            self.angle_ = normalized_angle
            self.valid_ = True
            return self.angle_

        # 计算角度差
        diff = self._computeAngleDiff(normalized_angle, self.angle_)

        # 限幅：单帧最大变化角度
        abs_diff = abs(diff)
        if abs_diff > self.max_step_deg_:
            diff = self.max_step_deg_ if diff > 0 else -self.max_step_deg_
            abs_diff = self.max_step_deg_

        # 计算自适应 alpha
        effective_alpha = self._computeAdaptiveAlpha(abs_diff)

        # 更新平滑角度
        self.angle_ = self._normalizeAngle(self.angle_ + effective_alpha * diff)

        return self.angle_

    def Reset(self) -> None:
        """重置平滑器状态"""
        self.angle_ = None
        self.valid_ = False

    @property
    def current_angle(self) -> Optional[float]:
        """获取当前平滑后的角度"""
        return self.angle_ if self.valid_ else None

    @property
    def is_valid(self) -> bool:
        """是否有有效的角度值"""
        return self.valid_
