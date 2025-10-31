#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析脚本
统计收集的游戏数据情况
"""

import json
import os
from pathlib import Path
import numpy as np
from collections import Counter
# import matplotlib.pyplot as plt  # 可选依赖

def analyze_session(session_path):
    """分析单个session的数据"""
    actions_file = session_path / "actions.json"
    frames_dir = session_path / "frames"
    
    if not actions_file.exists():
        return None
    
    # 加载动作数据
    with open(actions_file, 'r', encoding='utf-8') as f:
        actions = json.load(f)
    
    # 统计信息
    stats = {
        'session_id': session_path.name,
        'total_frames': len(actions),
        'frame_files': len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0,
        'key_counts': Counter(),
        'mouse_positions': [],
        'timestamps': []
    }
    
    # 分析每个动作
    for action in actions:
        # 按键统计
        for key in action.get('keys', []):
            stats['key_counts'][key] += 1
        
        # 鼠标位置
        mouse_pos = action.get('mouse_pos', [1920, 1080])
        stats['mouse_positions'].append(mouse_pos)
        
        # 时间戳
        stats['timestamps'].append(action.get('timestamp', 0))
    
    # 计算鼠标位置范围
    if stats['mouse_positions']:
        mouse_positions = np.array(stats['mouse_positions'])
        stats['mouse_x_range'] = [mouse_positions[:, 0].min(), mouse_positions[:, 0].max()]
        stats['mouse_y_range'] = [mouse_positions[:, 1].min(), mouse_positions[:, 1].max()]
    
    # 计算时长
    if stats['timestamps']:
        stats['duration'] = max(stats['timestamps']) - min(stats['timestamps'])
    
    return stats

def main():
    """主函数"""
    data_root = Path("../data_collection/data/recordings")
    
    if not data_root.exists():
        print(f"数据目录不存在: {data_root}")
        return
    
    print("=== 坦克世界数据统计 ===\n")
    
    # 分析所有session
    sessions = list(data_root.glob("session_*"))
    all_stats = []
    
    for session_path in sessions:
        stats = analyze_session(session_path)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        print("未找到有效的数据session")
        return
    
    # 总体统计
    total_frames = sum(s['total_frames'] for s in all_stats)
    total_duration = sum(s.get('duration', 0) for s in all_stats)
    
    print(f"总session数: {len(all_stats)}")
    print(f"总帧数: {total_frames}")
    print(f"总时长: {total_duration:.1f} 秒")
    print(f"平均FPS: {total_frames/total_duration:.1f}")
    
    # 按键统计
    all_key_counts = Counter()
    for stats in all_stats:
        all_key_counts.update(stats['key_counts'])
    
    print(f"\n=== 按键使用统计 ===")
    for key, count in all_key_counts.most_common():
        percentage = count / total_frames * 100
        print(f"{key}: {count} 次 ({percentage:.1f}%)")
    
    # 鼠标位置统计
    all_mouse_x = []
    all_mouse_y = []
    for stats in all_stats:
        if 'mouse_positions' in stats:
            mouse_positions = np.array(stats['mouse_positions'])
            all_mouse_x.extend(mouse_positions[:, 0])
            all_mouse_y.extend(mouse_positions[:, 1])
    
    if all_mouse_x:
        print(f"\n=== 鼠标位置统计 ===")
        print(f"X坐标范围: {min(all_mouse_x)} - {max(all_mouse_x)}")
        print(f"Y坐标范围: {min(all_mouse_y)} - {max(all_mouse_y)}")
        print(f"X坐标中心: {np.mean(all_mouse_x):.1f}")
        print(f"Y坐标中心: {np.mean(all_mouse_y):.1f}")
    
    # 各session详情
    print(f"\n=== 各Session详情 ===")
    for stats in all_stats:
        print(f"\n{stats['session_id']}:")
        print(f"  帧数: {stats['total_frames']}")
        print(f"  时长: {stats.get('duration', 0):.1f} 秒")
        print(f"  文件数: {stats['frame_files']}")
        if 'mouse_x_range' in stats:
            print(f"  鼠标X范围: {stats['mouse_x_range']}")
            print(f"  鼠标Y范围: {stats['mouse_y_range']}")
        
        # 最常用按键
        top_keys = stats['key_counts'].most_common(3)
        if top_keys:
            print(f"  常用按键: {', '.join([f'{k}({c})' for k, c in top_keys])}")

if __name__ == "__main__":
    main()
