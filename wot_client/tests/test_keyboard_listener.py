"""
测试键盘监听功能
Test keyboard listener functionality
"""

from pynput import keyboard
from loguru import logger
import sys
import ctypes

def is_admin():
    """检查是否有管理员权限"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def test_keyboard_listener():
    """测试键盘监听"""
    
    logger.info("=" * 80)
    logger.info("键盘监听测试工具")
    logger.info("=" * 80)
    
    # 检查管理员权限
    if is_admin():
        logger.info("✓ 当前以管理员权限运行")
    else:
        logger.warning("✗ 当前未以管理员权限运行")
        logger.warning("  在 Windows 上，pynput 需要管理员权限才能监听全局键盘")
        logger.warning("  请右键点击脚本，选择'以管理员身份运行'")
        logger.warning("  或者在管理员权限的命令提示符中运行")
        logger.warning("")
        logger.warning("  如果继续测试失败，请以管理员身份重新运行")
        logger.warning("")
        response = input("是否继续测试？(y/n): ").strip().lower()
        if response != 'y':
            return
    
    logger.info("")
    logger.info("测试说明：")
    logger.info("  1. 按下任意键，应该能看到按键被记录")
    logger.info("  2. 按 ESC 键退出测试")
    logger.info("  3. 如果按键没有被记录，说明权限不足")
    logger.info("=" * 80)
    logger.info("")
    
    pressed_keys = set()
    total_presses = 0
    
    def on_press(key):
        nonlocal total_presses
        total_presses += 1
        
        try:
            key_str = key.char
            pressed_keys.add(key_str)
            logger.info(f"✓ 按键按下: '{key_str}' | 当前按键: {pressed_keys} | 总计: {total_presses}")
        except AttributeError:
            key_str = str(key).replace('Key.', '')
            pressed_keys.add(key_str)
            logger.info(f"✓ 特殊键按下: '{key_str}' | 当前按键: {pressed_keys} | 总计: {total_presses}")
    
    def on_release(key):
        try:
            key_str = key.char
            pressed_keys.discard(key_str)
            logger.info(f"  释放: '{key_str}' | 当前按键: {pressed_keys}")
        except AttributeError:
            key_str = str(key).replace('Key.', '')
            pressed_keys.discard(key_str)
            logger.info(f"  释放: '{key_str}' | 当前按键: {pressed_keys}")
        
        # ESC 退出
        if key == keyboard.Key.esc:
            logger.info("")
            logger.info("=" * 80)
            logger.info("测试结束")
            logger.info(f"总按键次数: {total_presses}")
            if total_presses == 0:
                logger.error("❌ 没有捕获到任何按键！")
                logger.error("")
                logger.error("可能的原因：")
                logger.error("  1. 未以管理员权限运行")
                logger.error("  2. 安全软件阻止了键盘监听")
                logger.error("  3. pynput 安装有问题")
                logger.error("")
                logger.error("解决方案：")
                logger.error("  1. 右键点击脚本 -> '以管理员身份运行'")
                logger.error("  2. 检查安全软件设置")
                logger.error("  3. 重新安装 pynput: pip install --force-reinstall pynput")
            else:
                logger.info("✓ 键盘监听功能正常！")
                logger.info("  可以开始录制了")
            logger.info("=" * 80)
            return False
    
    logger.info("开始监听键盘输入...")
    logger.info("（现在可以按任意键测试）")
    logger.info("")
    
    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except Exception as e:
        logger.error(f"键盘监听失败: {e}")
        logger.error("")
        logger.error("请确保：")
        logger.error("  1. 以管理员权限运行")
        logger.error("  2. pynput 已正确安装: pip install pynput")
        return

if __name__ == "__main__":
    test_keyboard_listener()

