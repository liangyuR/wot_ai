#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于动态窗口算法（DWA）的小地图路径规划Demo
从地图图片中提取障碍物、起点和终点，使用DWA算法规划路径
"""

import math
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 导入二值化器
try:
    from path_planning.core.minimap_binarizer import MinimapBinarizer
except ImportError:
    from core.minimap_binarizer import MinimapBinarizer

# 复用demo.py中的DWA函数
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import (
    dwa_control, calc_dynamic_window, calc_control_and_trajectory,
    motion, predict_trajectory, calc_obstacle_cost, calc_to_goal_cost,
    plot_arrow, plot_robot, RobotType, Config
)

show_animation = True


class MinimapProcessor:
    """小地图处理器：从地图图片中提取障碍物、起点和终点"""
    
    def __init__(self, pixel_to_meter: float = 0.1, use_kmeans: bool = True, kmeans_clusters: int = 5):
        """
        初始化地图处理器
        
        Args:
            pixel_to_meter: 像素到米的转换比例（默认1像素=0.1米）
            use_kmeans: 是否使用KMeans进行二值化（默认True）
            kmeans_clusters: KMeans聚类数量（默认5）
        """
        self.pixel_to_meter_ = pixel_to_meter
        self.use_kmeans_ = use_kmeans
        self.map_image_ = None
        self.map_height_ = 0
        self.map_width_ = 0
        
        # 初始化二值化器
        if self.use_kmeans_:
            self.binarizer_ = MinimapBinarizer(n_clusters=kmeans_clusters)
        else:
            self.binarizer_ = None
    
    def LoadMap(self, map_path: str) -> bool:
        """
        加载地图图片
        
        Args:
            map_path: 地图图片路径
        
        Returns:
            是否加载成功
        """
        try:
            map_file = Path(map_path)
            if not map_file.exists():
                print(f"地图文件不存在: {map_path}")
                return False
            
            self.map_image_ = cv2.imread(str(map_file))
            if self.map_image_ is None:
                print(f"无法读取地图文件: {map_path}")
                return False
            
            self.map_height_, self.map_width_ = self.map_image_.shape[:2]
            print(f"地图加载成功: {self.map_width_}x{self.map_height_}")
            return True
        except Exception as e:
            print(f"加载地图失败: {e}")
            return False
    
    
    def ExtractObstacles(self) -> np.ndarray:
        """
        提取障碍物点
        如果使用KMeans，则基于KMeans二值化结果；否则使用灰度阈值方法
        
        Returns:
            障碍物点数组 [x(m), y(m), ...]
        """
        if self.map_image_ is None:
            return np.array([])
        
        # 使用KMeans二值化或传统方法
        if self.use_kmeans_ and self.binarizer_ is not None:
            # 使用KMeans二值化获取障碍物掩码
            obstacle_mask = self.binarizer_.GetObstacleMask(self.map_image_)
        else:
            # 传统方法：转为灰度图，阈值处理
            gray = cv2.cvtColor(self.map_image_, cv2.COLOR_BGR2GRAY)
            _, obstacle_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学处理，去除孤立小噪声
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找障碍物轮廓
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 采样障碍物点（每隔一定距离采样轮廓点）
        obstacle_points = []
        sample_interval = 10  # 每10像素采样1个点
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # 过滤非常小的区域
                continue
            
            for i in range(0, len(cnt), sample_interval):
                point = cnt[i][0]
                x_pixel = point[0]
                y_pixel = point[1]
                # 转为世界坐标，y轴需翻转
                x_meter = x_pixel * self.pixel_to_meter_
                y_meter = (self.map_height_ - y_pixel) * self.pixel_to_meter_
                obstacle_points.append([x_meter, y_meter])
        
        if obstacle_points:
            obstacles = np.array(obstacle_points)
            print(f"提取到 {len(obstacles)} 个障碍物点")
            return obstacles
        else:
            return np.array([]).reshape(0, 2)
    
    def ExtractStartPosition(self) -> Optional[Tuple[float, float, float]]:
        """
        提取起点位置（白色箭头）
        
        Returns:
            (x(m), y(m), yaw(rad)) 或 None
        """
        if self.map_image_ is None:
            return None
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(self.map_image_, cv2.COLOR_BGR2HSV)
        
        # 检测白色区域（箭头）
        # 白色在HSV中：S接近0，V接近255
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("未检测到白色箭头")
            return None
        
        # 过滤轮廓（箭头应该有一定面积和宽高比）
        arrow_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 1500:  # 箭头面积范围
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                if 1.1 < aspect_ratio < 3.5:  # 箭头宽高比
                    arrow_contours.append(cnt)
        
        if not arrow_contours:
            print("未找到符合条件的箭头")
            return None
        
        # 选择面积最大的箭头
        cnt = max(arrow_contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            return None
        
        # 计算箭头中心
        cx_pixel = int(M["m10"] / M["m00"])
        cy_pixel = int(M["m01"] / M["m00"])
        
        # 计算箭头朝向（找出离中心最远的点作为箭头尖端）
        cnt_points = cnt.reshape(-1, 2)
        tip = max(cnt_points, key=lambda p: np.hypot(p[0] - cx_pixel, p[1] - cy_pixel))
        
        # 计算角度（图像坐标系：Y向下）
        dx = tip[0] - cx_pixel
        dy = tip[1] - cy_pixel
        angle_rad = math.atan2(dy, dx)
        
        # 转换为世界坐标
        # 注意：图像坐标系Y向下，世界坐标系Y向上，需要翻转Y坐标
        x_meter = cx_pixel * self.pixel_to_meter_
        y_meter = (self.map_height_ - cy_pixel) * self.pixel_to_meter_
        
        # 角度也需要调整：图像坐标系Y向下，世界坐标系Y向上
        # 在图像坐标系中，Y向下，所以角度需要取反
        angle_rad = -angle_rad
        
        print(f"检测到起点: ({x_meter:.2f}, {y_meter:.2f}), 朝向: {math.degrees(angle_rad):.1f}°")
        return (x_meter, y_meter, angle_rad)
    
    def ExtractGoalPosition(self) -> Optional[Tuple[float, float]]:
        """
        提取终点位置（红色圆圈中的白色旗帜）
        
        Returns:
            (x(m), y(m)) 或 None
        """
        if self.map_image_ is None:
            return None
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(self.map_image_, cv2.COLOR_BGR2HSV)
        
        # 检测红色区域（敌方基地圆圈）
        # 红色在HSV中跨越两个范围：0-10 和 170-180
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 120])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找红色轮廓（圆形）
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("未检测到红色圆圈")
            return None
        
        # 查找最大的圆形轮廓（基地圆圈应该比较大）
        largest_contour = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area and area > 500:  # 基地圆圈应该比较大
                # 检查圆形度
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # 接近圆形
                        max_area = area
                        largest_contour = cnt
        
        if largest_contour is None:
            print("未找到符合条件的红色圆圈")
            return None
        
        # 计算圆心
        M = cv2.moments(largest_contour)
        if M["m00"] <= 0:
            return None
        
        cx_pixel = int(M["m10"] / M["m00"])
        cy_pixel = int(M["m01"] / M["m00"])
        
        # 转换为世界坐标
        # 注意：图像坐标系Y向下，世界坐标系Y向上，需要翻转
        x_meter = cx_pixel * self.pixel_to_meter_
        y_meter = (self.map_height_ - cy_pixel) * self.pixel_to_meter_
        
        print(f"检测到终点: ({x_meter:.2f}, {y_meter:.2f})")
        return (x_meter, y_meter)
    
    def GetMapImage(self) -> Optional[np.ndarray]:
        """获取地图图片（用于可视化）"""
        return self.map_image_
    
    def GetMapImageWithObstacles(self, obstacles: np.ndarray) -> Optional[np.ndarray]:
        """
        获取带有障碍物标记的地图图片（障碍物区域涂黑）
        
        Args:
            obstacles: 障碍物点数组 [x(m), y(m), ...]
        
        Returns:
            处理后的地图图片
        """
        if self.map_image_ is None:
            return self.map_image_
        
        # 复制地图图像
        map_with_obstacles = self.map_image_.copy()
        
        if len(obstacles) == 0:
            return map_with_obstacles
        
        # 重新提取障碍物掩码（与ExtractObstacles中的逻辑一致）
        if self.use_kmeans_ and self.binarizer_ is not None:
            # 使用KMeans二值化获取障碍物掩码
            obstacle_mask = self.binarizer_.GetObstacleMask(self.map_image_)
        else:
            # 传统方法
            gray = cv2.cvtColor(self.map_image_, cv2.COLOR_BGR2GRAY)
            _, obstacle_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学处理，去除孤立小噪声
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        
        # 将障碍物区域涂黑
        # 在掩码为白色的地方（障碍物），将地图图像涂黑
        map_with_obstacles[obstacle_mask > 0] = [0, 0, 0]
        
        return map_with_obstacles


def main():
    """主函数"""
    print("DWA路径规划Demo开始")
    
    # 1. 初始化地图处理器
    processor = MinimapProcessor(pixel_to_meter=0.1)
    
    # 2. 加载地图
    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "maps" / "01.png"
    
    if not processor.LoadMap(str(map_path)):
        print("地图加载失败，退出")
        return
    
    # 3. 提取障碍物
    obstacles = processor.ExtractObstacles()
    if len(obstacles) == 0:
        print("警告：未提取到障碍物，使用空障碍物数组")
        obstacles = np.array([]).reshape(0, 2)
    
    # 4. 提取起点
    start_info = processor.ExtractStartPosition()
    if start_info is None:
        print("无法检测起点，使用默认位置")
        start_info = (0.0, 0.0, 0.0)
    
    start_x, start_y, start_yaw = start_info
    
    # 5. 提取终点
    goal_pos = processor.ExtractGoalPosition()
    if goal_pos is None:
        print("无法检测终点，使用默认位置")
        goal_pos = (10.0, 10.0)
    
    goal = np.array(goal_pos)
    
    # 6. 初始化DWA配置
    config = Config()
    config.robot_type = RobotType.circle
    config.robot_radius = 2.0  # 根据地图尺寸调整
    config.max_speed = 2.0
    config.min_speed = -0.5
    config.max_yaw_rate = 40.0 * math.pi / 180.0
    config.max_accel = 0.3
    config.max_delta_yaw_rate = 40.0 * math.pi / 180.0
    config.v_resolution = 0.05
    config.yaw_rate_resolution = 0.2 * math.pi / 180.0
    config.dt = 0.1
    config.predict_time = 3.0
    config.to_goal_cost_gain = 0.15
    config.speed_cost_gain = 1.0
    config.obstacle_cost_gain = 1.0
    config.robot_stuck_flag_cons = 0.001
    config.ob = obstacles
    
    # 7. 初始化机器人状态 [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([start_x, start_y, start_yaw, 0.0, 0.0])
    
    # 8. 运行DWA路径规划
    trajectory = np.array(x)
    max_iterations = 1000
    iteration = 0
    
    print(f"开始路径规划: 起点({start_x:.2f}, {start_y:.2f}) -> 终点({goal[0]:.2f}, {goal[1]:.2f})")
    
    # 获取地图图片用于可视化（包含障碍物标记）
    map_image = processor.GetMapImageWithObstacles(obstacles)
    pixel_to_meter = processor.pixel_to_meter_
    
    if show_animation:
        plt.figure(figsize=(12, 10))
        # 显示地图背景
        if map_image is not None:
            # 计算地图在世界坐标系中的范围
            map_width_m = map_image.shape[1] * pixel_to_meter
            map_height_m = map_image.shape[0] * pixel_to_meter
            # 翻转Y轴以匹配matplotlib坐标系
            map_image_flipped = np.flipud(map_image)
            plt.imshow(map_image_flipped, extent=[0, map_width_m, 0, map_height_m],
                      alpha=0.3, origin='lower')
    
    while iteration < max_iterations:
        u, predicted_trajectory = dwa_control(x, config, goal, obstacles)
        x = motion(x, u, config.dt)
        trajectory = np.vstack((trajectory, x))
        
        if show_animation:
            plt.cla()
            # 显示地图背景
            if map_image is not None:
                map_width_m = map_image.shape[1] * pixel_to_meter
                map_height_m = map_image.shape[0] * pixel_to_meter
                map_image_flipped = np.flipud(map_image)
                plt.imshow(map_image_flipped, extent=[0, map_width_m, 0, map_height_m],
                          alpha=0.3, origin='lower')
            
            # 障碍物已在地图上涂黑，不需要再绘制点
            
            # 绘制预测轨迹
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g", linewidth=1, alpha=0.5)
            
            # 绘制当前轨迹
            plt.plot(trajectory[:, 0], trajectory[:, 1], "-r", linewidth=2, label="路径")
            
            # 绘制当前位置
            plt.plot(x[0], x[1], "xr", markersize=10, label="当前位置")
            
            # 绘制目标点
            plt.plot(goal[0], goal[1], "xb", markersize=15, label="目标")
            
            # 绘制机器人
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2], length=1.0, width=0.3)
            
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.title(f"DWA路径规划 - 迭代 {iteration}")
            plt.pause(0.01)
        
        # 检查是否到达目标
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius * 2:
            print(f"到达目标！迭代次数: {iteration}")
            break
        
        iteration += 1
        
        # 检查是否卡住（速度接近0且距离目标较远）
        if abs(x[3]) < 0.01 and dist_to_goal > config.robot_radius * 5:
            print(f"机器人可能卡住，停止规划。迭代次数: {iteration}")
            break
    
    print(f"路径规划完成，总迭代次数: {iteration}")
    
    if show_animation:
        # 最终显示完整路径
        plt.cla()
        if map_image is not None:
            map_width_m = map_image.shape[1] * pixel_to_meter
            map_height_m = map_image.shape[0] * pixel_to_meter
            map_image_flipped = np.flipud(map_image)
            plt.imshow(map_image_flipped, extent=[0, map_width_m, 0, map_height_m],
                      alpha=0.3, origin='lower')
        
        # 障碍物已在地图上涂黑，不需要再绘制点
        
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r", linewidth=2, label="最终路径")
        plt.plot(start_x, start_y, "og", markersize=10, label="起点")
        plt.plot(goal[0], goal[1], "xb", markersize=15, label="目标")
        plt.plot(x[0], x[1], "xr", markersize=10, label="当前位置")
        
        plot_robot(x[0], x[1], x[2], config)
        plot_arrow(x[0], x[1], x[2], length=1.0, width=0.3)
        
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.title("DWA路径规划结果")
        plt.show()


if __name__ == '__main__':
    main()

