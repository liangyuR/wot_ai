"""
磁盘性能测试工具
诊断磁盘写入速度是否足以支持实时录制
"""

import cv2
import numpy as np
import time
from pathlib import Path
import shutil


def test_disk_write_speed(output_dir="data/disk_test", duration=10):
    """
    测试磁盘写入速度
    
    Args:
        output_dir: 测试目录
        duration: 测试时长（秒）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🔍 磁盘性能测试")
    print("=" * 80)
    print(f"测试目录: {output_path.absolute()}")
    print(f"测试时长: {duration} 秒")
    print()
    
    # 生成测试帧（1920x1080 RGB）
    print("生成测试帧...")
    test_frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    print(f"帧大小: {test_frame.shape}")
    print()
    
    # 测试不同 JPEG 质量
    qualities = [95, 85, 75]
    
    for quality in qualities:
        print(f"测试 JPEG 质量 {quality}...")
        
        # 编码测试
        start_time = time.time()
        encoded = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        encode_time = time.time() - start_time
        file_size = len(encoded) / 1024  # KB
        
        print(f"  编码时间: {encode_time*1000:.1f} ms")
        print(f"  文件大小: {file_size:.1f} KB")
        
        # 磁盘写入测试
        frames_written = 0
        total_bytes = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            frame_path = output_path / f"test_q{quality}_{frames_written:06d}.jpg"
            cv2.imwrite(str(frame_path), test_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            total_bytes += frame_path.stat().st_size
            frames_written += 1
        
        elapsed = time.time() - start_time
        fps = frames_written / elapsed
        mb_per_sec = (total_bytes / 1024 / 1024) / elapsed
        
        print(f"  写入帧数: {frames_written}")
        print(f"  平均 FPS: {fps:.1f}")
        print(f"  写入速度: {mb_per_sec:.2f} MB/s")
        print(f"  总数据: {total_bytes / 1024 / 1024:.1f} MB")
        
        # 判断是否足够
        if fps >= 30:
            print(f"  ✓ 性能充足（目标 30 FPS）")
        elif fps >= 20:
            print(f"  ⚠ 性能勉强（建议降低质量或帧率）")
        else:
            print(f"  ❌ 性能不足（建议使用 SSD 或降低帧率）")
        print()
    
    # 清理测试文件
    print("清理测试文件...")
    shutil.rmtree(output_path)
    print("✓ 测试完成")
    print()
    
    # 建议
    print("=" * 80)
    print("💡 建议")
    print("=" * 80)
    print("如果性能不足，可以尝试：")
    print("  1. 使用 SSD 而非机械硬盘")
    print("  2. 降低 JPEG 质量（--jpeg-quality 85 或 75）")
    print("  3. 增加帧采样间隔（--frame-step 2 或 5）")
    print("  4. 降低录制帧率（--fps 20 或 15）")
    print("  5. 确保测试目录不在 USB 驱动器上")
    print()
    print("异步保存器已启用，可缓冲 2 秒的帧以平滑性能波动")
    print("=" * 80)


def test_async_performance():
    """测试异步保存器性能"""
    try:
        from data_collection.async_frame_saver import AsyncFrameSaver
    except ImportError:
        print("❌ 异步帧保存器不可用")
        return
    
    print("\n" + "=" * 80)
    print("🧵 异步保存器测试")
    print("=" * 80)
    
    output_dir = Path("data/disk_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建异步保存器
    saver = AsyncFrameSaver(output_dir, jpeg_quality=95, queue_size=60)
    saver.Start()
    
    # 生成测试帧
    test_frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    
    # 快速添加 60 帧
    print("快速添加 60 帧到队列...")
    start_time = time.time()
    for i in range(60):
        saver.SaveFrame(test_frame, i)
    add_time = time.time() - start_time
    
    print(f"添加时间: {add_time*1000:.1f} ms")
    print(f"平均每帧: {add_time/60*1000:.2f} ms")
    print(f"队列大小: {saver.GetQueueSize()}/60")
    
    # 等待保存完成
    print("\n等待队列清空...")
    start_wait = time.time()
    saver.Stop()
    wait_time = time.time() - start_wait
    
    stats = saver.GetStats()
    print(f"等待时间: {wait_time:.2f} 秒")
    print(f"已保存: {stats['saved']}")
    print(f"丢弃: {stats['dropped']}")
    print(f"平均保存速度: {stats['saved']/wait_time:.1f} FPS")
    
    if stats['saved'] >= 60 and stats['dropped'] == 0:
        print("✓ 异步保存器工作正常")
    else:
        print("⚠ 异步保存器可能有问题")
    
    # 清理
    shutil.rmtree(output_dir)
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    else:
        duration = 10
    
    print("磁盘性能诊断工具")
    print("此工具会测试您的磁盘是否足以支持实时录制")
    print()
    
    test_disk_write_speed(duration=duration)
    test_async_performance()

