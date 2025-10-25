"""
测试新的 session 数据格式
验证录制后的数据结构是否正确
"""

import json
import sys
from pathlib import Path

def test_session_format(session_dir: str):
    """测试 session 目录的数据格式"""
    session_path = Path(session_dir)
    
    if not session_path.exists():
        print(f"❌ Session 目录不存在: {session_dir}")
        return False
    
    print("=" * 80)
    print("🔍 验证 Session 数据格式")
    print("=" * 80)
    print(f"Session 目录: {session_path}")
    print()
    
    # 1. 检查必需文件
    print("1. 检查必需文件...")
    meta_path = session_path / "meta.json"
    actions_path = session_path / "actions.json"
    frames_dir = session_path / "frames"
    
    if not meta_path.exists():
        print("  ❌ meta.json 不存在")
        return False
    print("  ✓ meta.json 存在")
    
    if not actions_path.exists():
        print("  ❌ actions.json 不存在")
        return False
    print("  ✓ actions.json 存在")
    
    # 2. 验证 meta.json
    print("\n2. 验证 meta.json...")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        required_fields = [
            "session_id", "timestamp", "fps", "total_frames",
            "duration_seconds", "resolution", "capture_mode",
            "save_format", "frame_step", "frames_saved"
        ]
        
        missing_fields = [field for field in required_fields if field not in meta]
        if missing_fields:
            print(f"  ❌ 缺少字段: {missing_fields}")
            return False
        
        print(f"  ✓ 所有必需字段存在")
        print(f"    - Session ID: {meta['session_id']}")
        print(f"    - 时间戳: {meta['timestamp']}")
        print(f"    - FPS: {meta['fps']}")
        print(f"    - 总帧数: {meta['total_frames']}")
        print(f"    - 时长: {meta['duration_seconds']:.2f}s")
        print(f"    - 分辨率: {meta['resolution']}")
        print(f"    - 捕获模式: {meta['capture_mode']}")
        print(f"    - 保存格式: {meta['save_format']}")
        print(f"    - 帧采样: 每 {meta['frame_step']} 帧")
        print(f"    - 已保存帧: {meta['frames_saved']}")
        
    except json.JSONDecodeError as e:
        print(f"  ❌ meta.json 格式错误: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 读取 meta.json 失败: {e}")
        return False
    
    # 3. 验证 actions.json
    print("\n3. 验证 actions.json...")
    try:
        with open(actions_path, "r", encoding="utf-8") as f:
            actions = json.load(f)
        
        if not isinstance(actions, list):
            print(f"  ❌ actions.json 不是数组")
            return False
        
        print(f"  ✓ 加载成功，共 {len(actions)} 条记录")
        
        # 检查前几条记录的格式
        if len(actions) > 0:
            first_action = actions[0]
            required_fields = ["frame", "timestamp", "keys", "mouse_pos"]
            missing_fields = [field for field in required_fields if field not in first_action]
            
            if missing_fields:
                print(f"  ❌ 操作记录缺少字段: {missing_fields}")
                return False
            
            print(f"  ✓ 操作记录格式正确")
            print(f"    示例: frame={first_action['frame']}, keys={first_action['keys']}, mouse={first_action['mouse_pos']}")
            
            # 检查帧号是否连续
            if len(actions) > 1:
                frame_nums = [a['frame'] for a in actions[:10]]
                is_sequential = all(frame_nums[i] == i for i in range(len(frame_nums)))
                if is_sequential:
                    print(f"  ✓ 帧号连续")
                else:
                    print(f"  ⚠ 帧号不连续（可能正常）")
        
    except json.JSONDecodeError as e:
        print(f"  ❌ actions.json 格式错误: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 读取 actions.json 失败: {e}")
        return False
    
    # 4. 验证 frames/ 目录
    print("\n4. 验证 frames/ 目录...")
    if meta['save_format'] in ["frames", "both"]:
        if not frames_dir.exists():
            print(f"  ❌ frames/ 目录不存在")
            return False
        
        frame_files = list(frames_dir.glob("frame_*.jpg"))
        print(f"  ✓ frames/ 目录存在，共 {len(frame_files)} 个文件")
        
        # 验证帧数
        expected_frames = meta['frames_saved']
        if len(frame_files) != expected_frames:
            print(f"  ⚠ 警告: 预期 {expected_frames} 帧，实际 {len(frame_files)} 帧")
        else:
            print(f"  ✓ 帧数匹配")
        
        # 检查第一帧是否存在
        first_frame = frames_dir / "frame_000000.jpg"
        if first_frame.exists():
            file_size = first_frame.stat().st_size / 1024  # KB
            print(f"  ✓ 第一帧存在: {file_size:.1f} KB")
        else:
            print(f"  ❌ 第一帧不存在")
            return False
        
        # 验证帧文件名格式
        if len(frame_files) > 0:
            sample_name = frame_files[0].name
            import re
            if re.match(r'frame_\d{6}\.jpg', sample_name):
                print(f"  ✓ 帧文件名格式正确")
            else:
                print(f"  ❌ 帧文件名格式错误: {sample_name}")
                return False
    else:
        print(f"  ⊘ 跳过（save_format={meta['save_format']}，不保存帧）")
    
    # 5. 验证视频文件（可选）
    print("\n5. 验证视频文件（可选）...")
    video_path = session_path / "gameplay.avi"
    if meta['save_format'] in ["video", "both"]:
        if video_path.exists():
            file_size = video_path.stat().st_size / 1024 / 1024  # MB
            print(f"  ✓ 视频文件存在: {file_size:.1f} MB")
        else:
            print(f"  ❌ 视频文件不存在")
            return False
    else:
        if video_path.exists():
            print(f"  ⚠ 警告: 视频文件存在但 save_format={meta['save_format']}")
        else:
            print(f"  ⊘ 跳过（save_format={meta['save_format']}，不保存视频）")
    
    # 6. 数据完整性检查
    print("\n6. 数据完整性检查...")
    
    # 检查 actions 数量与 total_frames 是否匹配
    if len(actions) != meta['total_frames']:
        print(f"  ⚠ 警告: 操作记录数 ({len(actions)}) != 总帧数 ({meta['total_frames']})")
    else:
        print(f"  ✓ 操作记录数与总帧数匹配")
    
    # 检查持续时间计算
    expected_duration = meta['total_frames'] / meta['fps']
    if abs(meta['duration_seconds'] - expected_duration) > 0.1:
        print(f"  ⚠ 警告: 持续时间不一致 ({meta['duration_seconds']:.2f}s vs {expected_duration:.2f}s)")
    else:
        print(f"  ✓ 持续时间计算正确")
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("✅ Session 数据格式验证通过！")
    print("=" * 80)
    print(f"📊 统计:")
    print(f"  - 总帧数: {meta['total_frames']}")
    print(f"  - 时长: {meta['duration_seconds']:.2f}s")
    print(f"  - 平均 FPS: {meta['total_frames'] / meta['duration_seconds']:.2f}")
    if meta['save_format'] in ["frames", "both"]:
        print(f"  - 保存的帧: {meta['frames_saved']}")
        total_size = sum(f.stat().st_size for f in frames_dir.glob("*.jpg")) / 1024 / 1024
        print(f"  - 帧总大小: {total_size:.1f} MB")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python test_session_format.py <session_directory>")
        print("\n示例:")
        print("  python test_session_format.py data/sessions/session_20251025_154532")
        print("\n或测试最新的 session:")
        
        sessions_dir = Path("data/sessions")
        if sessions_dir.exists():
            sessions = sorted(sessions_dir.glob("session_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if sessions:
                print(f"  python test_session_format.py {sessions[0]}")
                print(f"\n自动测试最新 session: {sessions[0].name}")
                test_session_format(str(sessions[0]))
            else:
                print("\n  没有找到任何 session")
        else:
            print("\n  data/sessions 目录不存在")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    success = test_session_format(session_dir)
    sys.exit(0 if success else 1)

