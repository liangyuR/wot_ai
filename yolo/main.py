#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO detection script for gameplay recordings.
Processes frames from C:\data\recordings and runs object detection.
"""

import argparse
from pathlib import Path
from typing import List, Optional
import logging
import os

from ultralytics import YOLO
from PIL import Image
import cv2


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get script directory as base path
SCRIPT_DIR = Path(__file__).resolve().parent


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
    
    args = parser.parse_args()
    
    # Initialize detector with script directory as base
    detector = YoloDetector(model_path=args.model, base_dir=SCRIPT_DIR)
    detector.LoadModel()
    
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
