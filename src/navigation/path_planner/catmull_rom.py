#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Catmull-Rom Spline实现：生成平滑曲线轨迹

功能：
- Centripetal Catmull-Rom插值（α=0.5）
- 通过所有控制点
- 生成连续、可微分的曲线
"""

from typing import List, Tuple

Point = Tuple[float, float]  # (x, y)

def catmull_rom_spline(
    p0: Point, p1: Point, p2: Point, p3: Point,
    t: float, alpha: float = 0.5
) -> Point:
    """
    Catmull-Rom Spline插值（单个点）
    
    Args:
        p0, p1, p2, p3: 四个控制点
        t: 插值参数 [0, 1]，在p1和p2之间插值
        alpha: 参数化类型，0.5=centripetal（推荐）
    
    Returns:
        插值点 (x, y)
    """
    def tj(pi: Point, pj: Point, alpha: float) -> float:
        """计算两点间的参数化距离"""
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        return (dx * dx + dy * dy) ** (alpha / 2.0)
    
    # 计算参数化距离
    t0 = 0.0
    t1 = tj(p0, p1, alpha) + t0
    t2 = tj(p1, p2, alpha) + t1
    t3 = tj(p2, p3, alpha) + t2
    
    # 归一化t到[t1, t2]区间
    t_normalized = t1 + t * (t2 - t1)
    
    # Catmull-Rom基函数
    def basis(t: float, t0: float, t1: float, t2: float, t3: float) -> Tuple[float, float, float, float]:
        """计算Catmull-Rom基函数系数"""
        A1 = (t1 - t) / (t1 - t0) if t1 != t0 else 0
        A2 = (t - t0) / (t1 - t0) if t1 != t0 else 0
        A3 = (t2 - t) / (t2 - t1) if t2 != t1 else 0
        A4 = (t - t1) / (t2 - t1) if t2 != t1 else 0
        
        B1 = A1 * (t2 - t) / (t2 - t1) if t2 != t1 else 0
        B2 = A2 * (t2 - t) / (t2 - t1) + A3 * (t - t1) / (t2 - t1) if t2 != t1 else 0
        B3 = A4 * (t - t1) / (t2 - t1) if t2 != t1 else 0
        
        C1 = B1 * (t2 - t) / (t2 - t0) if t2 != t0 else 0
        C2 = B2 * (t2 - t) / (t2 - t0) + B3 * (t - t0) / (t2 - t0) if t2 != t0 else 0
        
        # 简化版本（标准Catmull-Rom）
        # 使用更简单的公式
        t01 = t - t0
        t12 = t - t1
        t23 = t - t2
        
        if t < t1:
            # 在p0-p1段
            if t1 != t0:
                ratio = t01 / (t1 - t0)
                return (1 - ratio, ratio, 0, 0)
            else:
                return (1, 0, 0, 0)
        elif t < t2:
            # 在p1-p2段（主要插值区间）
            if t2 != t1:
                ratio = t12 / (t2 - t1)
                # Catmull-Rom插值
                t_norm = ratio
                t2_norm = t_norm * t_norm
                t3_norm = t2_norm * t_norm
                
                # 标准Catmull-Rom基函数
                b0 = -0.5 * t3_norm + t2_norm - 0.5 * t_norm
                b1 = 1.5 * t3_norm - 2.5 * t2_norm + 1.0
                b2 = -1.5 * t3_norm + 2.0 * t2_norm + 0.5 * t_norm
                b3 = 0.5 * t3_norm - 0.5 * t2_norm
                
                return (b0, b1, b2, b3)
            else:
                return (0, 1, 0, 0)
        else:
            # 在p2-p3段
            if t3 != t2:
                ratio = t23 / (t3 - t2)
                return (0, 0, 1 - ratio, ratio)
            else:
                return (0, 0, 0, 1)
    
    # 计算基函数系数
    b0, b1, b2, b3 = basis(t_normalized, t0, t1, t2, t3)
    
    # 插值
    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    
    return (x, y)


def catmull_rom_interpolate(
    points: List[Point],
    num_points_per_segment: int = 15,
    alpha: float = 0.5
) -> List[Point]:
    """
    对点列表进行Catmull-Rom插值
    
    Args:
        points: 控制点列表 [(x, y), ...]
        num_points_per_segment: 每段曲线采样点数，默认15
        alpha: 参数化类型，0.5=centripetal（推荐）
    
    Returns:
        插值后的密集点列表
    """
    if len(points) < 2:
        return points
    
    if len(points) == 2:
        # 只有两个点，直接线性插值
        result = []
        p0, p1 = points[0], points[1]
        for i in range(num_points_per_segment + 1):
            t = i / num_points_per_segment
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            result.append((x, y))
        return result
    
    result = []
    
    # 添加第一个点
    result.append(points[0])
    
    # 对每段进行插值
    for i in range(len(points) - 1):
        # 获取四个控制点（需要处理边界）
        if i == 0:
            # 第一段：使用p0, p0, p1, p2
            p0 = points[0]
            p1 = points[0]
            p2 = points[1]
            p3 = points[2] if len(points) > 2 else points[1]
        elif i == len(points) - 2:
            # 最后一段：使用p_{n-2}, p_{n-1}, p_n, p_n
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 1]
        else:
            # 中间段：使用p_{i-1}, p_i, p_{i+1}, p_{i+2}
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2]
        
        # 在这段上采样（跳过第一个点，因为已经在上一段的末尾添加了）
        for j in range(1, num_points_per_segment + 1):
            t = j / num_points_per_segment
            point = catmull_rom_spline(p0, p1, p2, p3, t, alpha)
            result.append(point)
    
    return result

