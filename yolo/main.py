#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO detection script for gameplay recordings.
Processes frames from C:\data\recordings and runs object detection.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import os
import time
import yaml
import numpy as np

from ultralytics import YOLO
from PIL import Image
import cv2
import mss
from pynput import mouse, keyboard
from pynput.keyboard import Key, Listener as KeyboardListener

from aim_assist import AimAssist


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get script directory as base path
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "aim_config.yaml"


class YoloDetector:
    """YOLO object detector for gameplay frames."""
    
    def __init__(self, model_path: str = "model/yolo11m.pt", base_dir: Optional[Path] = None):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model file (relative or absolute).
            base_dir: Base directory for resolving relative paths (defaults to script directory).
        """
        if base_dir is None:
            base_dir = SCRIPT_DIR
        
        model_path_obj = Path(model_path)
        if model_path_obj.is_absolute():
            self.model_path_ = str(model_path_obj)
        else:
            # Resolve relative to base directory
            self.model_path_ = str(base_dir / model_path_obj)
        
        self.model_ = None
    
    def LoadModel(self):
        """Load YOLO model."""
        model_path_obj = Path(self.model_path_)
        if not model_path_obj.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path_}\n"
                f"Resolved path: {model_path_obj.absolute()}"
            )
        
        logger.info(f"Loading model from {model_path_obj.absolute()}")
        self.model_ = YOLO(str(model_path_obj))
        logger.info("Model loaded successfully")
    
    def DetectFrame(self, image_path: str, save_result: bool = False, 
                    output_dir: Optional[str] = None) -> dict:
        """
        Detect objects in a single frame.
        
        Args:
            image_path: Path to image file.
            save_result: Whether to save annotated result.
            output_dir: Output directory for results.
        
        Returns:
            Detection results dictionary.
        """
        if self.model_ is None:
            raise RuntimeError("Model not loaded. Call LoadModel() first.")
        
        results = self.model_(image_path)
        
        if save_result and output_dir:
            output_path = Path(output_dir) / Path(image_path).name
            results[0].save(str(output_path))
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detections.append({
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                })
        
        return {
            'image_path': image_path,
            'detections': detections,
            'count': len(detections)
        }
    
    def ProcessDirectory(self, directory: str, output_dir: Optional[str] = None,
                        max_frames: Optional[int] = None) -> List[dict]:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images.
            output_dir: Output directory for annotated results.
            max_frames: Maximum number of frames to process (None for all).
        
        Returns:
            List of detection results.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        image_files = sorted(list(dir_path.glob("*.jpg")) + 
                           list(dir_path.glob("*.png")))
        
        if max_frames:
            image_files = image_files[:max_frames]
        
        logger.info(f"Processing {len(image_files)} images from {directory}")
        
        results = []
        for idx, image_file in enumerate(image_files):
            if (idx + 1) % 10 == 0:
                logger.info(f"Processing frame {idx + 1}/{len(image_files)}")
            
            try:
                result = self.DetectFrame(
                    str(image_file),
                    save_result=(output_dir is not None),
                    output_dir=output_dir
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
        
        return results
    
    def ProcessRecordings(self, recordings_dir: str, 
                         output_base_dir: Optional[str] = None,
                         max_sessions: Optional[int] = None,
                         max_frames_per_session: Optional[int] = None):
        """
        Process all recording sessions.
        
        Args:
            recordings_dir: Base directory containing recording sessions.
            output_base_dir: Base output directory for results.
            max_sessions: Maximum number of sessions to process.
            max_frames_per_session: Maximum frames per session.
        """
        recordings_path = Path(recordings_dir)
        if not recordings_path.exists():
            raise ValueError(f"Recordings directory does not exist: {recordings_dir}")
        
        session_dirs = sorted([d for d in recordings_path.iterdir() 
                              if d.is_dir() and d.name.startswith("session_")])
        
        if max_sessions:
            session_dirs = session_dirs[:max_sessions]
        
        logger.info(f"Found {len(session_dirs)} recording sessions")
        
        all_results = []
        for session_idx, session_dir in enumerate(session_dirs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing session {session_idx + 1}/{len(session_dirs)}: {session_dir.name}")
            logger.info(f"{'='*60}")
            
            frames_dir = session_dir / "frames"
            if not frames_dir.exists():
                logger.warning(f"Frames directory not found: {frames_dir}")
                continue
            
            output_dir = None
            if output_base_dir:
                output_dir = Path(output_base_dir) / session_dir.name / "yolo_results"
                output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                results = self.ProcessDirectory(
                    str(frames_dir),
                    output_dir=str(output_dir) if output_dir else None,
                    max_frames=max_frames_per_session
                )
                
                # Summary
                total_detections = sum(r['count'] for r in results)
                logger.info(f"Session summary: {len(results)} frames, "
                          f"{total_detections} total detections")
                
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error processing session {session_dir.name}: {e}")
                continue
        
        # Overall summary
        logger.info(f"\n{'='*60}")
        logger.info("Overall Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total sessions processed: {len(session_dirs)}")
        logger.info(f"Total frames processed: {len(all_results)}")
        logger.info(f"Total detections: {sum(r['count'] for r in all_results)}")
        
        return all_results


def LoadAimConfig(config_path: Path) -> Optional[dict]:
    """加载瞄准辅助配置文件"""
    try:
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        logger.info("瞄准辅助配置加载成功")
        return config
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return None


def InitializeAimAssist(config: dict) -> Optional[AimAssist]:
    """初始化瞄准辅助系统"""
    try:
        screen_config = config.get('screen', {})
        fov_config = config.get('fov', {})
        mouse_config = config.get('mouse', {})
        
        screen_w = screen_config.get('width', 1920)
        screen_h = screen_config.get('height', 1080)
        h_fov = fov_config.get('horizontal', 90.0)
        v_fov = fov_config.get('vertical', None)
        mouse_sensitivity = mouse_config.get('sensitivity', 1.0)
        calibration_factor = mouse_config.get('calibration_factor', None)
        
        aim_assist = AimAssist(
            screen_width=screen_w,
            screen_height=screen_h,
            horizontal_fov=h_fov,
            vertical_fov=v_fov,
            mouse_sensitivity=mouse_sensitivity,
            calibration_factor=calibration_factor
        )
        
        logger.info(f"瞄准辅助系统初始化成功")
        return aim_assist
    except Exception as e:
        logger.error(f"初始化瞄准辅助系统失败: {e}")
        return None


def GetBestTarget(detections: List[dict], target_class: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """
    从检测结果中选择最佳目标
    
    Args:
        detections: 检测结果列表
        target_class: 目标类别ID（如果None则选择置信度最高的）
    
    Returns:
        目标坐标 (x, y)，如果未找到则返回None
    """
    if not detections:
        return None
    
    # 如果指定了类别，优先选择该类别的目标
    if target_class is not None:
        class_detections = [d for d in detections if d['class'] == target_class]
        if class_detections:
            detections = class_detections
    
    # 选择置信度最高的目标
    best_detection = max(detections, key=lambda x: x['confidence'])
    
    # 计算bbox中心
    bbox = best_detection['bbox']
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    
    return center_x, center_y


def RunRealtimeDetection(detector: YoloDetector, config_path: Path, fps: float = 30.0,
                         target_class: Optional[int] = None, enable_aim_assist: bool = True):
    """
    实时检测模式，支持瞄准辅助
    
    Args:
        detector: YOLO检测器
        config_path: 配置文件路径
        fps: 检测帧率
        target_class: 目标类别ID（如果None则选择置信度最高的）
        enable_aim_assist: 是否启用瞄准辅助
    """
    # 加载瞄准辅助配置
    aim_config = None
    aim_assist = None
    mouse_controller = None
    smoothing_config = None
    current_smoothed_x = 0.0
    current_smoothed_y = 0.0
    
    if enable_aim_assist:
        aim_config = LoadAimConfig(config_path)
        if aim_config:
            aim_assist = InitializeAimAssist(aim_config)
            if aim_assist:
                mouse_controller = mouse.Controller()
                smoothing_config = aim_config.get('smoothing', {})
                logger.info("瞄准辅助已启用")
            else:
                logger.warning("瞄准辅助初始化失败，将禁用瞄准辅助")
                enable_aim_assist = False
        else:
            logger.warning("无法加载瞄准辅助配置，将禁用瞄准辅助")
            enable_aim_assist = False
    
    # 屏幕捕获初始化
    sct = mss.mss()
    mon = sct.monitors[1]
    screen_w = mon['width']
    screen_h = mon['height']
    monitor = {"top": 0, "left": 0, "width": screen_w, "height": screen_h}
    logger.info(f"屏幕分辨率: {screen_w}x{screen_h}")
    
    # 热键控制
    enabled_flag = {'on': False}
    stop_flag = {'stop': False}
    
    def OnKeyPress(key):
        """键盘按下回调"""
        try:
            if key == Key.f8:
                enabled_flag['on'] = not enabled_flag['on']
                status = "启用" if enabled_flag['on'] else "禁用"
                logger.info(f"瞄准辅助: {status} (F8)")
            elif key == Key.esc:
                enabled_flag['on'] = False
                stop_flag['stop'] = True
                logger.info("退出实时检测模式 (ESC)")
        except Exception:
            pass
    
    keyboard_listener = KeyboardListener(on_press=OnKeyPress)
    keyboard_listener.start()
    
    # 主循环
    interval = 1.0 / fps
    frame_count = 0
    last_log_time = time.time()
    
    logger.info("="*60)
    logger.info("实时检测模式")
    logger.info("="*60)
    logger.info("热键控制:")
    logger.info("  F8 - 切换瞄准辅助启用/禁用")
    logger.info("  ESC - 退出")
    logger.info("="*60)
    
    try:
        while not stop_flag['stop']:
            frame_start_time = time.time()
            
            # 捕获屏幕
            frame_array = np.array(sct.grab(monitor))
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_BGRA2BGR)
            
            # 保存临时图像用于检测
            temp_image_path = SCRIPT_DIR / "temp_frame.jpg"
            cv2.imwrite(str(temp_image_path), frame_bgr)
            
            # 执行检测
            try:
                result_dict = detector.DetectFrame(str(temp_image_path), save_result=False)
                detections = result_dict.get('detections', [])
                
                # 如果启用瞄准辅助且有检测结果
                if enabled_flag['on'] and enable_aim_assist and detections:
                    target_pos = GetBestTarget(detections, target_class)
                    
                    if target_pos:
                        target_x, target_y = target_pos
                        
                        # 计算鼠标移动量
                        mouse_x, mouse_y = aim_assist.TargetToMouseMovement(
                            target_x, target_y
                        )
                        
                        # 平滑处理
                        smoothing_factor = smoothing_config.get('factor', 0.3)
                        max_step = smoothing_config.get('max_step', 50.0)
                        
                        dx, dy = aim_assist.SmoothMovement(
                            target_mouse_x=mouse_x,
                            target_mouse_y=mouse_y,
                            current_mouse_x=current_smoothed_x,
                            current_mouse_y=current_smoothed_y,
                            smoothing_factor=smoothing_factor,
                            max_step=max_step
                        )
                        
                        # 更新累计移动状态
                        current_smoothed_x += dx
                        current_smoothed_y += dy
                        
                        # 执行鼠标移动
                        mouse_controller.move_rel(int(dx), int(dy))
                        
                        if frame_count % 30 == 0:  # 每30帧打印一次
                            logger.debug(f"目标: ({target_x}, {target_y}), "
                                       f"鼠标移动: ({dx:.2f}, {dy:.2f})")
                
            except Exception as e:
                logger.error(f"检测失败: {e}")
            
            # 清理临时文件
            if temp_image_path.exists():
                temp_image_path.unlink()
            
            frame_count += 1
            
            # 定期日志输出
            current_time = time.time()
            if current_time - last_log_time >= 2.0:
                elapsed = current_time - frame_start_time
                actual_fps = 1.0 / elapsed if elapsed > 0 else 0
                logger.info(f"FPS: {actual_fps:.1f}, 帧数: {frame_count}, "
                          f"状态: {'启用' if enabled_flag['on'] else '禁用'}")
                last_log_time = current_time
            
            # 维持FPS
            elapsed = time.time() - frame_start_time
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        keyboard_listener.stop()
        logger.info("实时检测模式已退出")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="YOLO detection for gameplay recordings")
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default=r"C:\data\recordings",
        help="Directory containing recording sessions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/yolo11m.pt",
        help="Path to YOLO model file (relative to script directory or absolute)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for annotated results (optional)"
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Maximum number of sessions to process"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames per session to process"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: process only first session with limited frames"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image file to detect and display"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="实时检测模式（支持瞄准辅助）"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="实时检测模式下的帧率（默认30）"
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="目标类别ID（如果指定则只瞄准该类别）"
    )
    parser.add_argument(
        "--no-aim-assist",
        action="store_true",
        help="禁用瞄准辅助（仅在实时模式下有效）"
    )
    
    args = parser.parse_args()
    
    # Initialize detector with script directory as base
    detector = YoloDetector(model_path=args.model, base_dir=SCRIPT_DIR)
    detector.LoadModel()
    
    # Real-time detection mode
    if args.realtime:
        enable_aim_assist = not args.no_aim_assist
        RunRealtimeDetection(
            detector=detector,
            config_path=CONFIG_PATH,
            fps=args.fps,
            target_class=args.target_class,
            enable_aim_assist=enable_aim_assist
        )
        return
    
    # Single image detection mode
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image file not found: {args.image}")
            return
        
        logger.info(f"Detecting objects in: {args.image}")
        results = detector.model_(str(image_path))
        
        # Display results
        results[0].show()
        
        # Print detection details
        boxes = results[0].boxes
        logger.info(f"\nDetection Results:")
        logger.info(f"Total detections: {len(boxes)}")
        
        for idx, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            logger.info(f"  Detection {idx + 1}: class={cls}, confidence={conf:.3f}, "
                       f"bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        logger.info("\nDetection completed!")
        return
    
    # Test mode settings
    if args.test_mode:
        args.max_sessions = 1
        args.max_frames = 10
        logger.info("Running in test mode")
    
    # Process recordings
    try:
        results = detector.ProcessRecordings(
            recordings_dir=args.recordings_dir,
            output_base_dir=args.output_dir,
            max_sessions=args.max_sessions,
            max_frames_per_session=args.max_frames
        )
        
        logger.info("\nDetection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
