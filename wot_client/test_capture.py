"""
快速诊断脚本 - 测试屏幕捕获和输入模块
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from loguru import logger


def testCppBindings():
    """测试 C++ 绑定"""
    logger.info("\n" + "=" * 80)
    logger.info("测试 C++ 绑定模块")
    logger.info("=" * 80)
    
    try:
        from cpp_bindings import ScreenCapture, InputControl
        logger.info("✓ C++ 绑定导入成功")
        
        # 测试屏幕捕获
        try:
            sc = ScreenCapture(1920, 1080)
            logger.info("✓ ScreenCapture 初始化成功")
            
            buffer = sc.Capture()
            expected_size = 1920 * 1080 * 3
            actual_size = len(buffer)
            
            logger.info(f"✓ 屏幕捕获成功")
            logger.info(f"  - 捕获数据大小: {actual_size:,} 字节")
            logger.info(f"  - 预期大小: {expected_size:,} 字节")
            
            if actual_size == expected_size:
                logger.info("✓ 数据大小正确")
            else:
                logger.warning(f"⚠ 数据大小不匹配 (差异: {actual_size - expected_size} 字节)")
                
        except Exception as e:
            logger.error(f"✗ ScreenCapture 测试失败: {e}")
            import traceback
            traceback.print_exc()
            
        # 测试输入控制
        try:
            ic = InputControl()
            logger.info("✓ InputControl 初始化成功")
            logger.info("  (未执行实际操作以避免干扰)")
        except Exception as e:
            logger.error(f"✗ InputControl 测试失败: {e}")
            
        return True
        
    except ImportError as e:
        logger.error(f"✗ C++ 绑定导入失败: {e}")
        logger.error("\n可能的原因：")
        logger.error("  1. C++ 模块未编译")
        logger.error("  2. .pyd 文件不在 cpp_bindings 目录")
        logger.error("  3. 缺少运行时依赖（如 MSVC 运行时）")
        logger.error("\n解决方法：")
        logger.error("  - 运行: build_xmake.bat")
        logger.error("  - 确保 cpp_bindings.pyd 存在")
        return False


def testPythonFallback():
    """测试 Python fallback 模块"""
    logger.info("\n" + "=" * 80)
    logger.info("测试 Python Fallback 模块")
    logger.info("=" * 80)
    
    # 测试 mss
    try:
        import mss
        import numpy as np
        
        logger.info("✓ mss 模块导入成功")
        
        sct = mss.mss()
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)[:, :, :3]
        
        logger.info("✓ mss 屏幕捕获成功")
        logger.info(f"  - 捕获形状: {frame.shape}")
        logger.info(f"  - 数据类型: {frame.dtype}")
        
    except ImportError:
        logger.error("✗ mss 未安装")
        logger.error("  安装: pip install mss")
    except Exception as e:
        logger.error(f"✗ mss 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    # 测试 pynput
    try:
        from pynput import keyboard, mouse
        logger.info("✓ pynput 模块导入成功")
        logger.info("  (未执行实际操作以避免干扰)")
    except ImportError:
        logger.error("✗ pynput 未安装")
        logger.error("  安装: pip install pynput")
    except Exception as e:
        logger.error(f"✗ pynput 测试失败: {e}")


def testOpenCV():
    """测试 OpenCV"""
    logger.info("\n" + "=" * 80)
    logger.info("测试 OpenCV")
    logger.info("=" * 80)
    
    try:
        import cv2
        import numpy as np
        
        logger.info(f"✓ OpenCV 导入成功 (版本: {cv2.__version__})")
        
        # 测试视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        test_path = "test_video.mp4"
        out = cv2.VideoWriter(test_path, fourcc, 30, (1920, 1080))
        
        if out.isOpened():
            logger.info("✓ VideoWriter 初始化成功")
            
            # 写入一个测试帧
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            out.write(frame)
            out.release()
            
            # 清理测试文件
            import os
            if os.path.exists(test_path):
                os.remove(test_path)
                logger.info("✓ 视频写入测试成功")
        else:
            logger.error("✗ VideoWriter 打开失败")
            
    except ImportError:
        logger.error("✗ OpenCV 未安装")
        logger.error("  安装: pip install opencv-python")
    except Exception as e:
        logger.error(f"✗ OpenCV 测试失败: {e}")
        import traceback
        traceback.print_exc()


def testSystemInfo():
    """显示系统信息"""
    logger.info("\n" + "=" * 80)
    logger.info("系统信息")
    logger.info("=" * 80)
    
    import platform
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python 版本: {platform.python_version()}")
    logger.info(f"架构: {platform.machine()}")
    
    # 检查文件结构
    logger.info("\n文件结构检查：")
    
    cpp_bindings_dir = Path(__file__).parent / "cpp_bindings"
    if cpp_bindings_dir.exists():
        logger.info(f"✓ cpp_bindings 目录存在: {cpp_bindings_dir}")
        
        pyd_file = cpp_bindings_dir / "cpp_bindings.pyd"
        if pyd_file.exists():
            logger.info(f"✓ cpp_bindings.pyd 存在 (大小: {pyd_file.stat().st_size:,} 字节)")
        else:
            logger.warning(f"⚠ cpp_bindings.pyd 不存在")
            
        lib_file = cpp_bindings_dir / "cpp_bindings.lib"
        if lib_file.exists():
            logger.info(f"✓ cpp_bindings.lib 存在 (大小: {lib_file.stat().st_size:,} 字节)")
    else:
        logger.warning(f"⚠ cpp_bindings 目录不存在")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("🔍 WoT AI - 系统诊断工具")
    logger.info("=" * 80)
    
    testSystemInfo()
    
    cpp_ok = testCppBindings()
    testPythonFallback()
    testOpenCV()
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("📋 诊断总结")
    logger.info("=" * 80)
    
    if cpp_ok:
        logger.info("✓ C++ 加速模块可用 - 推荐使用")
    else:
        logger.warning("⚠ C++ 模块不可用 - 将使用 Python fallback")
        logger.warning("  性能会降低，建议编译 C++ 模块")
        
    logger.info("\n下一步：")
    logger.info("  1. 如果所有测试通过，运行: python record_main.py")
    logger.info("  2. 测试屏幕捕获: python record_main.py --test")
    logger.info("  3. 开始录制: python record_main.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

