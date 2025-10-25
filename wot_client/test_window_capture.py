"""
测试窗口捕获功能
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from loguru import logger
from utils.window_capture import listAllWindows, WindowCapture
import cv2
import time


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("🔍 窗口捕获测试工具")
    logger.info("=" * 80)
    
    # 列出所有窗口
    logger.info("\n正在枚举所有可见窗口...")
    windows = listAllWindows()
    
    logger.info(f"\n找到 {len(windows)} 个可见窗口:\n")
    logger.info(f"{'序号':<6}{'窗口标题':<50}{'进程名':<30}PID")
    logger.info("-" * 90)
    
    for i, w in enumerate(windows[:30], 1):  # 只显示前 30 个
        title = w['title'][:48] if len(w['title']) > 48 else w['title']
        process = w['process'][:28] if len(w['process']) > 28 else w['process']
        logger.info(f"{i:<6}{title:<50}{process:<30}{w['pid']}")
    
    if len(windows) > 30:
        logger.info(f"\n... 还有 {len(windows) - 30} 个窗口未显示")
    
    # 查找《坦克世界》相关窗口
    wot_windows = [w for w in windows if 'tank' in w['title'].lower() or 'wot' in w['title'].lower() or
                   'worldoftanks' in w['process'].lower()]
    
    if wot_windows:
        logger.info("\n" + "=" * 80)
        logger.info("✓ 找到《坦克世界》相关窗口：")
        for w in wot_windows:
            logger.info(f"  - {w['title']}")
            logger.info(f"    进程: {w['process']}, PID: {w['pid']}")
    else:
        logger.warning("\n未找到《坦克世界》相关窗口")
        logger.info("请确保游戏已启动")
    
    # 交互式测试
    logger.info("\n" + "=" * 80)
    logger.info("交互式测试")
    logger.info("=" * 80)
    
    print("\n选择测试方式：")
    print("  1. 通过进程名测试（输入进程名，如 'WorldOfTanks.exe'）")
    print("  2. 通过窗口标题测试（输入关键字，如 'chrome', 'notepad'）")
    print("  3. 退出")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == "1":
        process_name = input("输入进程名（如 'WorldOfTanks.exe'）: ").strip()
        if process_name:
            test_capture(process_name=process_name)
    elif choice == "2":
        window_title = input("输入窗口标题关键字: ").strip()
        if window_title:
            test_capture(window_title=window_title)
    else:
        logger.info("退出...")


def test_capture(process_name=None, window_title=None):
    """测试窗口捕获"""
    try:
        logger.info("\n" + "=" * 80)
        logger.info("开始测试窗口捕获...")
        logger.info("=" * 80)
        
        # 创建窗口捕获器
        if process_name:
            logger.info(f"通过进程名查找: {process_name}")
            wc = WindowCapture(process_name=process_name, partial_match=True)
        else:
            logger.info(f"通过窗口标题查找: {window_title}")
            wc = WindowCapture(window_title=window_title, partial_match=True)
        
        if not wc.IsValid():
            logger.error("✗ 窗口捕获器无效")
            return
        
        logger.info(f"\n窗口信息:")
        logger.info(f"  - 宽度: {wc.GetWidth()}")
        logger.info(f"  - 高度: {wc.GetHeight()}")
        logger.info(f"  - 位置: {wc.GetWindowRect()}")
        
        # 测试捕获
        logger.info("\n开始捕获测试（共 5 帧）...")
        for i in range(5):
            start_time = time.time()
            frame = wc.Capture()
            elapsed = (time.time() - start_time) * 1000
            
            if frame is not None:
                logger.info(f"✓ 帧 {i+1}/5: shape={frame.shape}, 耗时={elapsed:.2f}ms")
                
                # 保存第一帧
                if i == 0:
                    save_path = "test_window_capture.png"
                    cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    logger.info(f"  已保存到: {save_path}")
            else:
                logger.error(f"✗ 帧 {i+1}/5: 捕获失败")
            
            time.sleep(0.2)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ 测试完成！")
        logger.info("=" * 80)
        logger.info("\n如果测试成功，说明窗口捕获功能正常。")
        logger.info("可以使用以下命令开始录制：")
        if process_name:
            logger.info(f"  python record_main.py --mode window --process {process_name}")
        else:
            logger.info(f"  python record_main.py --mode window --window-title \"{window_title}\"")
        
    except Exception as e:
        logger.error(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

