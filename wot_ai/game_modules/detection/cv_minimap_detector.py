#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于传统CV方法的小地图检测器
使用颜色过滤和轮廓检测识别小地图元素（己方、队友、敌方、基地）
"""

from typing import Optional, Dict, Tuple, List
from pathlib import Path
import numpy as np
import cv2
from loguru import logger


class CvMinimapDetector:
    """基于传统CV方法的小地图检测器"""
    
    def __init__(self, minimap_region: Dict, color_filters: Optional[Dict] = None,
                 detect_config: Optional[Dict] = None):
        """
        初始化小地图检测器
        
        Args:
            minimap_region: 小地图区域配置
                - 绝对坐标: {'x': int, 'y': int, 'width': int, 'height': int}
                - 比例配置: {'x_ratio': float, 'y_ratio': float, 'width_ratio': float, 'height_ratio': float}
            color_filters: 颜色过滤配置字典，可选
            detect_config: 检测开关配置字典，可选
                {
                    'detect_self': bool,      # 是否检测己方位置（默认True）
                    'detect_allies': bool,    # 是否检测队友（默认True）
                    'detect_enemies': bool,   # 是否检测敌人（默认True）
                    'detect_ally_base': bool, # 是否检测我方基地（默认True）
                    'detect_enemy_base': bool # 是否检测敌方基地（默认True）
                }
        """
        self.minimap_region_ = minimap_region
        
        # 默认检测配置
        default_detect_config = {
            'detect_self': True,
            'detect_allies': True,
            'detect_enemies': True,
            'detect_ally_base': True,
            'detect_enemy_base': True
        }
        
        if detect_config:
            default_detect_config.update(detect_config)
        self.detect_config_ = default_detect_config
        
        # 默认颜色过滤配置
        # 敌方坦克颜色 BGR(225, 18, 0) 转换为 HSV
        enemy_tank_bgr = np.uint8([[[225, 18, 0]]])
        enemy_tank_hsv = cv2.cvtColor(enemy_tank_bgr, cv2.COLOR_BGR2HSV)[0][0]
        enemy_tank_lower = np.array([
            max(0, enemy_tank_hsv[0] - 10),
            max(0, enemy_tank_hsv[1] - 50),
            max(0, enemy_tank_hsv[2] - 30)
        ])
        enemy_tank_upper = np.array([
            min(180, enemy_tank_hsv[0] + 10),
            min(255, enemy_tank_hsv[1] + 50),
            min(255, enemy_tank_hsv[2] + 30)
        ])
        
        # 敌方基地颜色：使用暗红色范围（参考代码）
        # 暗红色在HSV中跨越两个范围：0-10 和 170-180
        enemy_base_lower1 = np.array([0, 60, 40])
        enemy_base_upper1 = np.array([10, 255, 255])
        enemy_base_lower2 = np.array([170, 60, 40])
        enemy_base_upper2 = np.array([180, 255, 255])
        
        default_filters = {
            'white': {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 30, 255])
            },
            'green': {
                'lower': np.array([40, 40, 40]),
                'upper': np.array([80, 255, 255])
            },
            'red': {
                # 红色在HSV中跨越两个范围：0-10 和 170-180
                'lower1': np.array([0, 120, 120]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([170, 120, 120]),
                'upper2': np.array([180, 255, 255])
            },
            'enemy': {
                'lower': enemy_tank_lower,
                'upper': enemy_tank_upper
            },
            'enemy_base': {
                # 暗红色在HSV中跨越两个范围
                'lower1': enemy_base_lower1,
                'upper1': enemy_base_upper1,
                'lower2': enemy_base_lower2,
                'upper2': enemy_base_upper2
            }
        }
        
        if color_filters:
            default_filters.update(color_filters)
        self.color_filters_ = default_filters
        
        # 形状筛选参数
        self.arrow_params_ = {
            'min_area': 50,
            'max_area': 1500,
            'min_aspect_ratio': 0.9,
            'max_aspect_ratio': 3.5,
            'min_vertices': 3,
            'max_vertices': 7
        }
        
        # 坦克类型检测参数
        self.tank_params_ = {
            'min_area': 30,
            'max_area': 2000,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 4.0,
            'min_vertices': 3,
            'max_vertices': 8
        }
        
        # 旗帜检测参数
        self.flag_params_ = {
            'min_area': 100,
            'max_area': 10000,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_vertices': 3,
            'max_vertices': 10
        }
    
    def ExtractMinimap(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从完整帧中提取小地图区域
        
        Args:
            frame: 完整屏幕帧（BGR格式）
        
        Returns:
            小地图区域（BGR格式），如果区域无效则返回 None
        """
        try:
            h, w = frame.shape[:2]
            
            # 检查是绝对坐标还是比例配置
            if 'x' in self.minimap_region_:
                x = self.minimap_region_['x']
                y = self.minimap_region_['y']
                width = self.minimap_region_['width']
                height = self.minimap_region_['height']
            elif 'x_ratio' in self.minimap_region_:
                x = int(w * self.minimap_region_['x_ratio'])
                y = int(h * self.minimap_region_['y_ratio'])
                width = int(w * self.minimap_region_['width_ratio'])
                height = int(h * self.minimap_region_['height_ratio'])
            else:
                logger.error("无效的小地图区域配置")
                return None
            
            # 边界检查
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = min(width, w - x)
            height = min(height, h - y)
            
            minimap = frame[y:y+height, x:x+width]
            return minimap
        except Exception as e:
            logger.error(f"提取小地图失败: {e}")
            return None

    def FilterByColor(self, minimap: np.ndarray, color_name: str, 
                     apply_morphology: bool = True) -> np.ndarray:
        """
        根据颜色名称过滤图像
        
        Args:
            minimap: 小地图图像（BGR格式）
            color_name: 颜色名称（'white', 'green', 'red', 'enemy', 'enemy_base'）
            apply_morphology: 是否应用形态学处理（默认True）
        
        Returns:
            二值化掩码
        """
        if color_name not in self.color_filters_:
            logger.warning(f"未知的颜色名称: {color_name}")
            return np.zeros(minimap.shape[:2], dtype=np.uint8)
        
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        filter_config = self.color_filters_[color_name]
        
        # 红色和敌方基地需要特殊处理（跨越两个范围）
        if (color_name == 'red' or color_name == 'enemy_base') and 'lower1' in filter_config:
            mask1 = cv2.inRange(hsv, filter_config['lower1'], filter_config['upper1'])
            mask2 = cv2.inRange(hsv, filter_config['lower2'], filter_config['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, filter_config['lower'], filter_config['upper'])
        
        # 形态学处理去噪（可选）
        if apply_morphology:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        return mask
    
    def FilterContours(self, contours: List, params: Dict) -> List:
        """
        根据参数过滤轮廓（优化：先按面积过滤，减少不必要的approxPolyDP计算）
        
        Args:
            contours: 轮廓列表
            params: 过滤参数字典
        
        Returns:
            过滤后的轮廓列表
        """
        filtered = []
        for idx, cnt in enumerate(contours):
            # 先按面积过滤（快速检查）
            area = cv2.contourArea(cnt)
            if area < params['min_area'] or area > params['max_area']:
                logger.debug(f"轮廓 {idx}: 面积 {area:.2f} 不在范围 [{params['min_area']}, {params['max_area']}]，被过滤")
                continue
            
            # 再检查宽高比（快速检查）
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            if not (params['min_aspect_ratio'] < aspect_ratio < params['max_aspect_ratio']):
                logger.debug(
                    f"轮廓 {idx}: 宽高比 {aspect_ratio:.2f} 不在范围 ({params['min_aspect_ratio']}, {params['max_aspect_ratio']})，被过滤"
                )
                continue
            
            # 最后计算approxPolyDP（较慢，但已过滤大部分）
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vertices_count = len(approx)
            
            if params['min_vertices'] <= vertices_count <= params['max_vertices']:
                logger.debug(f"轮廓 {idx}: 通过过滤 (面积: {area:.2f}，宽高比: {aspect_ratio:.2f}，顶点数: {vertices_count})")
                filtered.append(cnt)
            else:
                logger.debug(
                    f"轮廓 {idx}: 顶点数 {vertices_count} 不在范围 [{params['min_vertices']}, {params['max_vertices']}]，被过滤"
                )
        
        logger.debug(f"总轮廓数: {len(contours)}，过滤后: {len(filtered)}")
        return filtered
    
    def GetContourCenter(self, cnt: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        计算轮廓质心
        
        Args:
            cnt: 轮廓
        
        Returns:
            (x, y) 或 None
        """
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def ClassifyTankType(self, cnt: np.ndarray, minimap: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        识别坦克类型
        
        Args:
            cnt: 轮廓
            minimap: 小地图图像
            bbox: 边界框 (x, y, w, h)
        
        Returns:
            坦克类型: 'light', 'heavy', 'medium', 'td', 'spg', 'unknown'
        """
        x, y, w_box, h_box = bbox
        
        # 提取该区域的ROI用于分析
        roi = minimap[max(0, y):min(minimap.shape[0], y+h_box), 
                      max(0, x):min(minimap.shape[1], x+w_box)]
        if roi.size == 0:
            return 'unknown'
        
        # 计算轮廓特征
        area = cv2.contourArea(cnt)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        
        # 多边形近似
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices_count = len(approx)
        
        # 检查是否为倒置三角形（TD）
        if vertices_count == 3:
            # 检查三角形是否倒置（尖端向下）
            cnt_points = cnt.reshape(-1, 2)
            y_coords = cnt_points[:, 1]
            # 如果最低点接近轮廓中心，可能是倒置三角形
            cy = np.mean(y_coords)
            bottom_points = cnt_points[y_coords > cy + (np.max(y_coords) - cy) * 0.6]
            if len(bottom_points) == 1:
                return 'td'
            # 否则可能是普通三角形（可能是其他类型）
        
        # 检查是否为正方形（火炮）
        if vertices_count == 4:
            # 计算宽高比，接近1:1可能是正方形
            if 0.8 < aspect_ratio < 1.2:
                # 检查是否为规则四边形
                if len(approx) == 4:
                    return 'spg'
        
        # 检查是否为菱形（轻型坦克）
        if vertices_count == 4:
            # 菱形通常宽高比接近1，但角度不同
            # 检查对角线长度
            if len(approx) == 4:
                pts = approx.reshape(-1, 2)
                # 计算对角线
                diag1 = np.hypot(pts[0][0] - pts[2][0], pts[0][1] - pts[2][1])
                diag2 = np.hypot(pts[1][0] - pts[3][0], pts[1][1] - pts[3][1])
                # 如果对角线长度相近，可能是菱形
                if diag1 > 0 and diag2 > 0 and abs(diag1 - diag2) / max(diag1, diag2) < 0.3:
                    if 0.7 < aspect_ratio < 1.4:
                        return 'light'
        
        # 检测多重条杠（重型和中型坦克）
        # 在ROI区域内查找平行矩形结构
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_binary = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)
        
        # 根据ROI的宽高比决定检测方向
        roi_aspect = w_box / h_box if h_box > 0 else 1
        
        if roi_aspect > 1.2:
            # 宽大于高，检测水平方向的条杠
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w_box // 3), 1))
            processed = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, horizontal_kernel)
            direction = 'horizontal'
        else:
            # 高大于等于宽，检测垂直方向的条杠
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, h_box // 3)))
            processed = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, vertical_kernel)
            direction = 'vertical'
        
        # 统计条杠数量
        bar_contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 统计显著的条杠（面积足够大，且长度合理）
        min_bar_area = area * 0.05
        bars = []
        for bar_cnt in bar_contours:
            bar_area = cv2.contourArea(bar_cnt)
            if bar_area > min_bar_area:
                bx, by, bw, bh = cv2.boundingRect(bar_cnt)
                bar_length = max(bw, bh)
                bar_width = min(bw, bh)
                # 条杠应该比较长且窄
                if bar_length > bar_width * 2 and bar_length > min(w_box, h_box) * 0.3:
                    bars.append(bar_cnt)
        
        total_bars = len(bars)
        
        if total_bars >= 3:
            return 'heavy'
        elif total_bars == 2:
            return 'medium'
        
        # 如果无法识别，返回unknown
        return 'unknown'
    
    def DetectTanks(self, minimap: np.ndarray, color_name: str) -> List[Dict]:
        """
        检测坦克位置和类型（统一处理队友和敌方）
        
        Args:
            minimap: 小地图图像（BGR格式）
            color_name: 颜色名称（'green' 或 'enemy'）
        
        Returns:
            坦克信息列表 [{'pos': (x, y), 'type': str}, ...]
        """
        mask = self.FilterByColor(minimap, color_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = self.FilterContours(contours, self.tank_params_)
        
        tanks = []
        for cnt in filtered:
            center = self.GetContourCenter(cnt)
            if center:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                tank_type = self.ClassifyTankType(cnt, minimap, (x, y, w_box, h_box))
                tanks.append({
                    'pos': center,
                    'type': tank_type
                })
        
        return tanks
    
    def DetectSelf(self, minimap: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """
        检测己方位置和朝向
        
        Args:
            minimap: 小地图图像（BGR格式）
        
        Returns:
            (x, y, angle) 或 None，其中 angle 为朝向角度（度）
        """
        mask = self.FilterByColor(minimap, 'white')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = self.FilterContours(contours, self.arrow_params_)
        
        if not filtered:
            return None
        
        # 选择面积最大的符合条件轮廓
        cnt = max(filtered, key=cv2.contourArea)
        center = self.GetContourCenter(cnt)
        if center is None:
            return None
        
        cx, cy = center
        
        # 计算箭头朝向（找出离中心最远的点作为箭头尖端）
        cnt_points = cnt.reshape(-1, 2)
        tip = max(cnt_points, key=lambda p: np.hypot(p[0] - cx, p[1] - cy))
        angle = np.degrees(np.arctan2(tip[1] - cy, tip[0] - cx))
        
        logger.debug(f"检测到己方位置: ({cx}, {cy}), 角度: {angle:.1f}°")
        return (cx, cy, angle)
    
    def DetectAllies(self, minimap: np.ndarray) -> List[Dict]:
        """
        检测队友坦克位置和类型
        
        Args:
            minimap: 小地图图像（BGR格式）
        
        Returns:
            队友信息列表 [{'pos': (x, y), 'type': str}, ...]
        """
        allies = self.DetectTanks(minimap, 'green')
        logger.debug(f"检测到 {len(allies)} 个队友")
        return allies
    
    def DetectEnemies(self, minimap: np.ndarray) -> List[Dict]:
        """
        检测敌方坦克位置和类型
        
        Args:
            minimap: 小地图图像（BGR格式）
        
        Returns:
            敌方信息列表 [{'pos': (x, y), 'type': str}, ...]
        """
        enemies = self.DetectTanks(minimap, 'enemy')
        logger.debug(f"检测到 {len(enemies)} 个敌方")
        return enemies
    
    def DetectFlagInRoi_(self, roi: np.ndarray) -> List[Dict]:
        """
        在ROI区域内检测白色旗帜
        
        Args:
            roi: ROI区域图像（BGR格式）
        
        Returns:
            白色旗帜候选列表 [{'pos': (x, y), 'area': float}, ...]
        """
        white_mask = self.FilterByColor(roi, 'white')
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_flags = self.FilterContours(contours, self.flag_params_)
        
        flag_candidates = []
        for cnt in filtered_flags:
            center = self.GetContourCenter(cnt)
            if center:
                area = cv2.contourArea(cnt)
                flag_candidates.append({
                    'pos': center,
                    'area': area
                })
        
        return flag_candidates
    
    def DetectBaseSingle_(self, minimap: np.ndarray, color_name: str) -> Optional[Dict]:
        """
        检测单个基地（假设地图上只有一个该颜色的圆形）
        参考代码：使用暗红色范围和面积比圆形度判断
        优化：移除重复形态学处理，按面积排序优先处理大轮廓
        
        Args:
            minimap: 小地图图像（BGR格式）
            color_name: 颜色名称（'green' 或 'enemy_base'）
        
        Returns:
            基地信息字典或None:
            {
                'center': (x, y),
                'radius': int,
                'bbox': (x0, y0, width, height),
                'flag_candidates': [{'pos': (x, y), 'area': float}, ...],
                'area': float
            }
        """
        # FilterByColor已经做了形态学处理，这里只需要额外的CLOSE操作（针对基地需要）
        color_mask = self.FilterByColor(minimap, color_name, apply_morphology=True)
        # 对基地检测，额外做CLOSE操作以连接断开的区域
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 优化：先按面积排序，优先处理大轮廓（基地通常较大）
        contours_with_area = [(cnt, cv2.contourArea(cnt)) for cnt in contours]
        contours_with_area.sort(key=lambda x: x[1], reverse=True)
        
        # 遍历轮廓，找到最佳圆形（参考代码）
        best_circle = None
        best_score = 0
        
        for cnt, area in contours_with_area:
            if area < 100:  # 小噪声过滤（参考代码）
                break  # 已排序，后续更小，直接退出
            
            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius ** 2)
            
            # 使用面积比计算圆形度（参考代码）
            if circle_area > 0:
                circularity = area / circle_area
            else:
                continue
            
            # 限制形状接近圆形（参考代码：0.6 < circularity < 1.2）
            if 0.6 < circularity < 1.2:
                # 进一步验证：基地的半径应该较大
                if radius >= 20:  # 半径太小，可能是坦克
                    # 选面积最大的圆
                    if area > best_score:
                        best_circle = (int(x), int(y), int(radius), area)
                        best_score = area
        
        if best_circle is None:
            return None
        
        x, y, radius, area = best_circle
        
        # ROI 包含圆形 + padding
        x0 = max(0, x - radius - 5)
        y0 = max(0, y - radius - 5)
        x1 = min(minimap.shape[1], x + radius + 5)
        y1 = min(minimap.shape[0], y + radius + 5)
        roi = minimap[y0:y1, x0:x1]
        
        # 检测旗帜
        flag_candidates = self.DetectFlagInRoi_(roi)
        
        logger.debug(f"检测到基地: ({x}, {y}), 半径: {radius}, 面积: {area:.0f}")
        
        return {
            'center': (x, y),
            'radius': radius,
            'bbox': (x0, y0, x1 - x0, y1 - y0),
            'flag_candidates': flag_candidates,
            'area': area
        }
    
    def DetectBases(self, minimap: np.ndarray) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        检测双方基地位置
        
        Args:
            minimap: 小地图图像（BGR格式）
        
        Returns:
            {'ally': (x, y) or None, 'enemy': (x, y) or None}
        """
        ally_base = None
        enemy_base = None
        
        # 检测白色旗帜，有两个，且一个是己方，一个是敌方。 当白色旗帜被绿色圆形包裹时，是己方基地，被红色包裹时，是对方基地。
        
        # 检测己方基地（绿色圆形）
        ally_base_info = self.DetectBaseSingle_(minimap, 'green')
        if ally_base_info:
            ally_base = ally_base_info['center']
            logger.debug(f"检测到己方基地: {ally_base}, 半径: {ally_base_info['radius']}, "
                        f"旗帜候选: {len(ally_base_info['flag_candidates'])}")
        
        # 检测敌方基地（使用敌方基地颜色）
        enemy_base_info = self.DetectBaseSingle_(minimap, 'enemy_base')
        if enemy_base_info:
            enemy_base = enemy_base_info['center']
            logger.debug(f"检测到敌方基地: {enemy_base}, 半径: {enemy_base_info['radius']}, "
                        f"旗帜候选: {len(enemy_base_info['flag_candidates'])}")

        logger.debug(f"己方基地: {ally_base}, 敌方基地: {enemy_base}")
        return {'ally': ally_base, 'enemy': enemy_base}
    
    def Parse(self, frame: np.ndarray) -> Dict:
        """
        解析完整帧，返回所有小地图元素
        
        Args:
            frame: 完整屏幕帧（BGR格式）
        
        Returns:
            结构化检测结果:
            {
                'self': {'pos': (x, y) or None, 'angle': float or None},
                'allies': [{'pos': (x, y), 'type': str}, ...],
                'enemies': [{'pos': (x, y), 'type': str}, ...],
                'bases': {'ally': (x, y) or None, 'enemy': (x, y) or None}
            }
            type 可能的值: 'light', 'heavy', 'medium', 'td', 'spg', 'unknown'
        """
        minimap = self.ExtractMinimap(frame)
        if minimap is None:
            return {
                'self': {'pos': None, 'angle': None},
                'allies': [],
                'enemies': [],
                'bases': {'ally': None, 'enemy': None}
            }
        
        # 根据配置开关检测各个元素
        self_result = None
        if self.detect_config_.get('detect_self', True):
            self_result = self.DetectSelf(minimap)
        
        allies = []
        if self.detect_config_.get('detect_allies', True):
            allies = self.DetectAllies(minimap)
        
        enemies = []
        if self.detect_config_.get('detect_enemies', True):
            enemies = self.DetectEnemies(minimap)
        
        bases = self.DetectBases(minimap)
        
        # 组合结果
        if self_result:
            self_pos = (self_result[0], self_result[1])
            self_angle = self_result[2]
        else:
            self_pos = None
            self_angle = None
        
        return {
            'self': {
                'pos': self_pos,
                'angle': self_angle
            },
            'allies': allies,
            'enemies': enemies,
            'bases': bases
        }
