#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图CV检测演示脚本：使用 CvMinimapDetector 模块检测并可视化小地图元素
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

try:
    from yolo.core.cv_minimap_detector import CvMinimapDetector
except ImportError:
    from core.cv_minimap_detector import CvMinimapDetector

try:
    from path_planning.core.minimap_binarizer import MinimapBinarizer
except ImportError:
    from core.minimap_binarizer import MinimapBinarizer

# 导入DWA相关函数
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "path_planning"))
from demo import (
    dwa_control, motion, Config, RobotType, plot_robot, plot_arrow
)


def DrawResults(minimap: np.ndarray, results: dict) -> np.ndarray:
    """
    在小地图上绘制检测结果
    
    Args:
        minimap: 小地图图像
        results: 检测结果字典
    
    Returns:
        绘制后的图像
    """
    display = minimap.copy()
    
    # 绘制己方位置
    if results['self']['pos']:
        cx, cy = results['self']['pos']
        angle = results['self']['angle']
        
        cv2.circle(display, (cx, cy), 8, (0, 255, 255), -1)  # 黄色实心圆
        cv2.circle(display, (cx, cy), 12, (0, 255, 255), 2)  # 黄色外圈
        
        # 绘制方向线
        if angle is not None:
            line_length = 100
            dx = np.cos(np.radians(angle)) * line_length
            dy = np.sin(np.radians(angle)) * line_length
            ex = int(cx + dx)
            ey = int(cy + dy)
            cv2.line(display, (cx, cy), (ex, ey), (0, 255, 255), 2)
        
        cv2.putText(display, "Self", (cx + 15, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 坦克类型标签映射
    type_labels = {
        'light': 'LT',
        'heavy': 'HT',
        'medium': 'MT',
        'td': 'TD',
        'spg': 'SPG',
        'unknown': '?'
    }
    
    # 绘制队友位置（绿色）
    for ally in results['allies']:
        cx, cy = ally['pos']
        tank_type = ally.get('type', 'unknown')
        cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
        cv2.circle(display, (cx, cy), 10, (0, 255, 0), 2)
        label = type_labels.get(tank_type, '?')
        cv2.putText(display, label, (cx + 12, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # 绘制敌方位置（红色）
    for enemy in results['enemies']:
        cx, cy = enemy['pos']
        tank_type = enemy.get('type', 'unknown')
        cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(display, (cx, cy), 10, (0, 0, 255), 2)
        label = type_labels.get(tank_type, '?')
        cv2.putText(display, label, (cx + 12, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 绘制己方基地（白色）
    if results['bases']['ally']:
        cx, cy = results['bases']['ally']
        cv2.rectangle(display, (cx - 10, cy - 10), (cx + 10, cy + 10), (255, 255, 255), 2)
        cv2.putText(display, "Ally Base", (cx + 15, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 绘制敌方基地（红色）
    if results['bases']['enemy']:
        cx, cy = results['bases']['enemy']
        cv2.rectangle(display, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 255), 2)
        cv2.putText(display, "Enemy Base", (cx + 15, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return display


def ExtractObstaclesFromBinary(binary_mask: np.ndarray, sample_interval: int = 5) -> np.ndarray:
    """
    从二值化图像中提取障碍物点（黑色区域）
    
    Args:
        binary_mask: 二值化图像（白色=道路，黑色=障碍物）
        sample_interval: 采样间隔（像素）
    
    Returns:
        障碍物点数组 [x, y, ...]（像素坐标）
    """
    # 找到黑色区域（值=0）作为障碍物
    obstacle_mask = (binary_mask == 0).astype(np.uint8) * 255
    
    # 形态学处理，去除孤立小噪声
    kernel = np.ones((3, 3), np.uint8)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
    
    # 查找障碍物轮廓
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 采样障碍物点
    obstacle_points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # 过滤非常小的区域
            continue
        
        # 每隔一定距离采样轮廓点
        for i in range(0, len(cnt), sample_interval):
            point = cnt[i][0]
            obstacle_points.append([float(point[0]), float(point[1])])
    
    if obstacle_points:
        return np.array(obstacle_points)
    else:
        return np.array([]).reshape(0, 2)


def main():
    """主函数"""
    # 读取截图
    img_path = "frame_000278.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    # 配置小地图区域（使用比例配置）
    h, w = img.shape[:2]
    minimap_region = {
        'x_ratio': 0.75,
        'y_ratio': 0.65,
        'width_ratio': 0.25,
        'height_ratio': 0.35
    }
    
    # 配置检测开关（可选，默认全部开启）
    detect_config = {
        'detect_self': True,        # 检测己方位置
        'detect_allies': False,     # 检测队友
        'detect_enemies': False,    # 检测敌人
        'detect_ally_base': False,  # 检测我方基地
        'detect_enemy_base': True   # 检测敌方基地
    }
    
    # 创建检测器
    detector = CvMinimapDetector(minimap_region, detect_config=detect_config)
    
    # 解析小地图
    results = detector.Parse(img)
    
    # 打印结果
    print("=" * 50)
    print("小地图检测结果:")
    print(f"己方位置: {results['self']['pos']}")
    if results['self']['angle'] is not None:
        print(f"己方朝向: {results['self']['angle']:.1f}°")
    print(f"队友数量: {len(results['allies'])}")
    for i, ally in enumerate(results['allies']):
        pos = ally['pos']
        tank_type = ally.get('type', 'unknown')
        type_name = {'light': '轻型', 'heavy': '重型', 'medium': '中型', 
                     'td': 'TD', 'spg': '火炮', 'unknown': '未知'}.get(tank_type, '未知')
        print(f"  队友 {i+1}: {pos} ({type_name})")
    print(f"敌方数量: {len(results['enemies'])}")
    for i, enemy in enumerate(results['enemies']):
        pos = enemy['pos']
        tank_type = enemy.get('type', 'unknown')
        type_name = {'light': '轻型', 'heavy': '重型', 'medium': '中型', 
                     'td': 'TD', 'spg': '火炮', 'unknown': '未知'}.get(tank_type, '未知')
        print(f"  敌方 {i+1}: {pos} ({type_name})")
    print(f"己方基地: {results['bases']['ally']}")
    print(f"敌方基地: {results['bases']['enemy']}")
    print("=" * 50)
    
    # 提取小地图
    minimap = detector.ExtractMinimap(img)
    if minimap is None:
        print("无法提取小地图区域")
        return
    
    # 检查检测结果有效性
    if not results['self']['pos']:
        print("错误：无法检测到己方位置")
        return
    
    if not results['bases']['enemy']:
        print("错误：无法检测到敌方基地位置")
        return
    
    # 提取障碍物（使用KMeans二值化）
    print("\n开始提取障碍物...")
    binarizer = MinimapBinarizer(n_clusters=3)
    binary_mask = binarizer.Binarize(minimap)  # 白色=道路，黑色=障碍物
    obstacles = ExtractObstaclesFromBinary(binary_mask, sample_interval=5)
    
    if len(obstacles) == 0:
        print("警告：未提取到障碍物，使用空障碍物数组")
        obstacles = np.array([]).reshape(0, 2)
    else:
        print(f"提取到 {len(obstacles)} 个障碍物点")
    
    # 获取起点和终点（像素坐标）
    self_pos = results['self']['pos']
    enemy_base_pos = results['bases']['enemy']
    self_angle = results['self']['angle']
    
    start_x = float(self_pos[0])
    start_y = float(self_pos[1])
    start_yaw = math.radians(self_angle) if self_angle is not None else 0.0
    
    goal = np.array([float(enemy_base_pos[0]), float(enemy_base_pos[1])])
    
    print(f"\n起点: ({start_x:.1f}, {start_y:.1f}), 朝向: {math.degrees(start_yaw):.1f}°")
    print(f"终点: ({goal[0]:.1f}, {goal[1]:.1f})")
    
    # 配置DWA参数（像素坐标）
    config = Config()
    config.robot_type = RobotType.circle
    config.robot_radius = 5.0  # 像素
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
    
    # 初始化机器人状态 [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([start_x, start_y, start_yaw, 0.0, 0.0])
    
    # 运行DWA路径规划
    print("\n开始路径规划...")
    trajectory = np.array(x)
    max_iterations = 1000
    iteration = 0
    
    # 初始化可视化（创建两个子图：主图和二值化图）
    plt.ion()
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1)  # 主图：路径规划
    ax_binary = fig.add_subplot(1, 2, 2)  # 二值化图
    
    while iteration < max_iterations:
        u, predicted_trajectory = dwa_control(x, config, goal, obstacles)
        x = motion(x, u, config.dt)
        trajectory = np.vstack((trajectory, x))
        
        # 可视化
        ax.clear()
        ax_binary.clear()
        
        # 主图：显示小地图背景
        minimap_rgb = cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB)
        ax.imshow(minimap_rgb, extent=[0, minimap.shape[1], minimap.shape[0], 0], alpha=0.7)
        
        # 显示可通行区域（白色区域）
        road_vis = np.zeros_like(binary_mask)
        road_vis[binary_mask == 255] = 255
        ax.imshow(road_vis, extent=[0, minimap.shape[1], minimap.shape[0], 0], 
                 cmap='Greens', alpha=0.2, vmin=0, vmax=255)
        
        # 显示障碍物区域（黑色区域）
        obstacle_vis = np.zeros_like(binary_mask)
        obstacle_vis[binary_mask == 0] = 255
        ax.imshow(obstacle_vis, extent=[0, minimap.shape[1], minimap.shape[0], 0], 
                 cmap='Reds', alpha=0.3, vmin=0, vmax=255)
        
        # 绘制预测轨迹
        if len(predicted_trajectory) > 0:
            ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g", 
                   linewidth=1, alpha=0.5, label="预测轨迹")
        
        # 绘制实际路径
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-r", linewidth=2, label="实际路径")
        
        # 绘制当前位置
        ax.plot(x[0], x[1], "xr", markersize=10, label="当前位置")
        
        # 绘制目标点
        ax.plot(goal[0], goal[1], "xb", markersize=15, label="目标（敌方基地）")
        
        # 绘制机器人（需要设置当前axes）
        plt.sca(ax)  # 设置当前axes为主图
        plot_robot(x[0], x[1], x[2], config)
        plot_arrow(x[0], x[1], x[2], length=10.0, width=2.0)
        
        ax.set_xlim(0, minimap.shape[1])
        ax.set_ylim(minimap.shape[0], 0)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title(f"DWA路径规划 - 迭代 {iteration}")
        
        # 二值化图：显示二值化结果
        ax_binary.imshow(binary_mask, extent=[0, minimap.shape[1], minimap.shape[0], 0], 
                        cmap='gray', vmin=0, vmax=255)
        ax_binary.set_xlim(0, minimap.shape[1])
        ax_binary.set_ylim(minimap.shape[0], 0)
        ax_binary.set_aspect('equal')
        ax_binary.grid(True)
        ax_binary.set_title("二值化结果（白色=可通行，黑色=障碍物）")
        
        plt.tight_layout()
        plt.pause(0.01)
        
        # 检查是否到达目标
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius * 2:
            print(f"到达目标！迭代次数: {iteration}")
            break
        
        iteration += 1
        
        # 检查是否卡住（速度接近0且距离目标较远）
        if abs(x[3]) < 0.01 and dist_to_goal > config.robot_radius * 3:
            print(f"机器人可能卡住，停止规划。迭代次数: {iteration}")
            break
    
    if iteration >= max_iterations:
        print(f"达到最大迭代次数: {max_iterations}")
    
    print("路径规划完成")
    plt.ioff()
    plt.show()
    
    # 同时显示OpenCV窗口（可选）
    minimap_display = DrawResults(minimap, results)
    cv2.imshow("Minimap with Detections", minimap_display)
    print("按任意键关闭OpenCV窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
