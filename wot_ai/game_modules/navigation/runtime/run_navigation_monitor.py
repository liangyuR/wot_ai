#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时小地图监控主程序

基于：
- YOLO 检测模块 (navigation/core/minimap_detector.py)
- A* 路径规划 (navigation/core/path_planner.py)
- 栅格地图建模 (navigation/core/map_modeler.py)
- 透明浮窗叠加显示 (navigation/runtime/transparent_overlay.py)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

# 统一导入机制
from wot_ai.utils.paths import setup_python_path
from wot_ai.utils.imports import try_import_multiple
setup_python_path()

SetupLogger = None
logger_module, _ = try_import_multiple([
    'wot_ai.game_modules.common.utils.logger',
    'game_modules.common.utils.logger',
    'common.utils.logger',
    'yolo.utils.logger'
])
if logger_module is not None:
    SetupLogger = getattr(logger_module, 'SetupLogger', None)

if SetupLogger is None:
    from ...common.utils.logger import SetupLogger

logger = SetupLogger(__name__)

# 导入导航监控器
from .navigation_monitor import NavigationMonitor


def ParseArgs():
    """
    解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="实时小地图导航监控系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_navigation_monitor.py \\
      --model "data/models/training/minimap/exp_minimap/weights/best.pt" \\
      --region "160,160" \\
      --perf high
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="YOLO 模型路径 (.pt)"
    )
    
    parser.add_argument(
        "--region", 
        type=str, 
        default="520,520",
        help="小地图区域尺寸 'width,height'，自动从屏幕右下角裁剪 (默认: 160,160)"
    )
    
    parser.add_argument(
        "--perf", 
        type=str, 
        choices=["high", "medium", "low"],
        default="high", 
        help="性能档位: high(30fps) / medium(15fps) / low(8fps) (默认: medium)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="基础目录路径（用于解析相对路径，默认: 项目根目录）"
    )
    
    return parser.parse_args()


def GetScreenSize() -> tuple:
    """
    获取屏幕尺寸
    
    Returns:
        (width, height) 屏幕宽度和高度
    """
    try:
        import mss
        sct = mss.mss()
        monitor = sct.monitors[1]  # 主显示器
        return (monitor['width'], monitor['height'])
    except Exception as e:
        logger.error(f"获取屏幕尺寸失败: {e}")
        # 返回默认值（1920x1080）
        logger.warning("使用默认屏幕尺寸: 1920x1080")
        return (1920, 1080)


def ParseRegion(region_str: str) -> Dict:
    """
    解析小地图区域字符串，自动从屏幕右下角计算位置
    
    Args:
        region_str: 区域字符串，格式为 "width,height"
    
    Returns:
        区域字典 {'x': int, 'y': int, 'width': int, 'height': int}
    
    Raises:
        SystemExit: 如果解析失败
    """
    try:
        parts = region_str.split(",")
        if len(parts) != 2:
            raise ValueError("需要2个参数: width,height")
        
        w, h = map(int, parts)
        
        if w <= 0 or h <= 0:
            raise ValueError("宽度和高度必须为正数")
        
        # 获取屏幕尺寸
        screen_width, screen_height = GetScreenSize()
        
        # 从右下角计算坐标（参考 cut_minimap.py 的逻辑）
        x = max(0, screen_width - w)
        y = max(0, screen_height - h)
        
        logger.info(f"屏幕尺寸: {screen_width}x{screen_height}")
        logger.info(f"小地图区域: ({x}, {y}) - {w}x{h}")
        
        return {"x": x, "y": y, "width": w, "height": h}
    except Exception as e:
        logger.error(f"无效的 region 参数: {region_str} (应为 width,height)")
        logger.error(f"错误详情: {e}")
        sys.exit(1)


def Main():
    """主函数：启动导航监控系统"""
    args = ParseArgs()
    
    # 验证模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path.absolute()}")
        sys.exit(1)
    
    # 解析小地图区域
    minimap_region = ParseRegion(args.region)
    
    # 解析基础目录
    base_dir = None
    if args.base_dir:
        base_dir_path = Path(args.base_dir)
        if not base_dir_path.exists():
            logger.warning(f"基础目录不存在: {base_dir_path}，将使用默认路径解析")
        else:
            base_dir = base_dir_path
    
    # 打印启动信息
    logger.info("=" * 60)
    logger.info("启动实时小地图监控系统")
    logger.info("=" * 60)
    logger.info(f"模型路径: {model_path.absolute()}")
    logger.info(f"小地图区域: {minimap_region}")
    logger.info(f"性能档位: {args.perf}")
    if base_dir:
        logger.info(f"基础目录: {base_dir.absolute()}")
    logger.info("=" * 60)
    
    try:
        # 初始化并启动监控器
        monitor = NavigationMonitor(
            model_path=str(model_path),
            minimap_region=minimap_region,
            performance=args.perf,
            base_dir=base_dir
        )
        
        logger.info("监控器初始化成功，开始运行...")
        logger.info("按 Ctrl+C 停止监控")
        logger.info("")
        
        # 运行主循环（阻塞）
        monitor.Run()
        
    except KeyboardInterrupt:
        logger.info("")
        logger.info("收到中断信号，正在停止监控...")
        if 'monitor' in locals():
            monitor.Stop()
        logger.info("监控已停止")
    except Exception as e:
        logger.error(f"监控系统运行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    Main()

