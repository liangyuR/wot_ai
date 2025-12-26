#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimap detector: detect self arrow and enemy flag, with optional failure frame dump.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.utils.angle_smoother import AngleSmoother
from src.utils.global_path import GetGlobalConfig
from src.vision.detection_engine import DetectionEngine
from src.vision.opencv_arrow_detector import OpencvArrowDetector


@dataclass
class MinimapDetectionResult:
    """Unified output for minimap detection."""

    self_pos: Optional[Tuple[float, float]]
    self_angle: Optional[float]
    enemy_flag_pos: Optional[Tuple[float, float]]
    raw_detections: List[Dict]


class MinimapDetector:
    """
    Dual-model minimap detector:
    1) detect_engine_: detect arrow + enemy flag on the full minimap
    2) pose_engine_: refine arrow pose (keypoints) after cropping
    Fallback to OpenCV arrow detector if pose detection fails.
    """

    def __init__(self, enable_tracking: bool = True):
        config = GetGlobalConfig()

        # Detection params
        self.conf_threshold_ = config.model.conf_threshold
        self.iou_threshold_ = config.model.iou_threshold
        self.crop_size_ = getattr(config.model, "crop_size", 36)
        self.min_keypoint_conf_ = getattr(config.model, "min_keypoint_conf", 0.5)

        # Tracking
        self.enable_tracking_ = enable_tracking
        self.tracker_type_ = getattr(config.model, "tracker", "bytetrack.yaml")

        # Engines
        self.detect_engine_ = DetectionEngine(config.model.base_path)
        self.pose_engine_ = DetectionEngine(config.model.arrow_path)
        self.arrow_detector_ = OpencvArrowDetector()

        # Angle smoother
        angle_cfg = config.angle_detection
        self.angle_smoother_ = AngleSmoother(
            alpha=angle_cfg.smoothing_alpha,
            max_step_deg=angle_cfg.max_step_deg,
            noise_threshold_deg=angle_cfg.noise_threshold_deg,
            normal_threshold_deg=angle_cfg.normal_threshold_deg,
            noise_alpha_factor=angle_cfg.noise_alpha_factor,
            large_turn_alpha_factor=angle_cfg.large_turn_alpha_factor,
        )

        # Cached positions
        self.base_position_: Optional[Tuple[float, float]] = None

        # Classes
        self.arrow_class_id_: int = config.model.class_id_arrow
        self.flag_class_id_: int = config.model.class_id_flag

        # Failure frame dump (for training hard samples)
        fail_cfg = getattr(config, "failure_dump", None)
        self.fail_capture_enabled_ = bool(getattr(fail_cfg, "enable", False)) if fail_cfg else False
        self.fail_capture_interval_s_ = float(getattr(fail_cfg, "interval_s", 10.0)) if fail_cfg else 10.0
        minimap_dir = getattr(fail_cfg, "minimap_dir", "Logs/fail_minimap") if fail_cfg else "Logs/fail_minimap"
        arrow_dir = getattr(fail_cfg, "arrow_dir", "Logs/fail_arrow") if fail_cfg else "Logs/fail_arrow"
        self.fail_minimap_dir_ = Path(minimap_dir)
        self.fail_arrow_dir_ = Path(arrow_dir)
        self._last_fail_save_ts = 0.0
        if self.fail_capture_enabled_:
            self.fail_minimap_dir_.mkdir(parents=True, exist_ok=True)
            self.fail_arrow_dir_.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    def LoadModel(self) -> bool:
        """Load detection and pose models."""
        if not self.detect_engine_.LoadModel():
            logger.error("MinimapDetector: detect_engine load failed")
            return False

        if not self.pose_engine_.LoadModel():
            logger.error("MinimapDetector: pose_engine load failed")
            return False

        logger.info("MinimapDetector: models loaded")
        return True

    def Detect(self, frame: np.ndarray) -> MinimapDetectionResult:
        """Detect targets on minimap image (BGR)."""
        if frame is None or frame.size == 0:
            logger.debug("MinimapDetector: empty frame")
            return MinimapDetectionResult(None, None, None, [])

        # Step 1: global detection
        if self.enable_tracking_:
            detect_result = self.detect_engine_.Track(
                frame,
                confidence_threshold=self.conf_threshold_,
                iou_threshold=self.iou_threshold_,
                max_det=10,
                persist=True,
                tracker=self.tracker_type_,
            )
        else:
            detect_result = self.detect_engine_.Detect(
                frame,
                confidence_threshold=self.conf_threshold_,
                iou_threshold=self.iou_threshold_,
                max_det=10,
            )

        if not detect_result or len(detect_result) == 0:
            logger.debug("MinimapDetector: detect_engine empty result")
            self._maybe_save_failure(frame, None, kind="minimap")
            return MinimapDetectionResult(None, None, self.base_position_, [])

        # Parse detections
        raw_detections = self._parseResults(detect_result)

        # Cache enemy flag position once
        if self.base_position_ is None:
            self.base_position_ = self._findDetectionCenter(detect_result, self.flag_class_id_)
            if self.base_position_ is not None:
                logger.info(f"enemy_flag detected and cached at {self.base_position_}")

        # Find self arrow bbox
        arrow_bbox = self._findDetectionBbox(detect_result, self.arrow_class_id_)
        if arrow_bbox is None:
            logger.debug("MinimapDetector: self_arrow not found")
            self._maybe_save_failure(frame, None, kind="minimap")
            return MinimapDetectionResult(None, None, self.base_position_, raw_detections)

        # Step 2: crop arrow region
        crop, crop_offset = self._cropArrowRegion(frame, arrow_bbox)
        if crop is None:
            self._maybe_save_failure(frame, None, kind="minimap")
            return MinimapDetectionResult(None, None, self.base_position_, raw_detections)

        # Step 3: detect arrow pose
        local_center, raw_angle = self._detectArrowPose(crop)
        if local_center is None:
            self._maybe_save_failure(frame, crop, kind="arrow")

        # Convert to global coords
        self_pos = None
        if local_center is not None:
            cx_local, cy_local = local_center
            self_pos = (cx_local + crop_offset[0], cy_local + crop_offset[1])

        # Smooth angle (AngleSmoother handles None)
        self_angle = self.angle_smoother_.Update(raw_angle)

        return MinimapDetectionResult(
            self_pos=self_pos,
            self_angle=self_angle,
            enemy_flag_pos=self.base_position_,
            raw_detections=raw_detections,
        )

    def Reset(self) -> None:
        """Reset detector state (including trackers)."""
        self.angle_smoother_.Reset()
        self.base_position_ = None

        if self.enable_tracking_:
            self.detect_engine_.ResetTracker()
            self.pose_engine_.ResetTracker()

    def resetSession(self) -> None:
        """Reset session-level state and clean CUDA cache.
        
        Call this at end of each game match to:
        1. Reset angle smoother and cached positions
        2. Clear tracker history (prevents memory accumulation)
        3. Clean CUDA cache (prevents VRAM leak)
        
        Note: This keeps the model loaded on GPU for fast restart.
        """
        # Reset state
        self.angle_smoother_.Reset()
        self.base_position_ = None
        
        # Reset trackers (clears accumulated track history)
        if self.enable_tracking_:
            self.detect_engine_.ResetTracker()
            self.pose_engine_.ResetTracker()
        
        # Clean CUDA cache
        self._cleanupCudaCache()
        
        logger.info("MinimapDetector: session reset complete")

    def _cleanupCudaCache(self) -> None:
        """Clean CUDA cache while keeping models loaded.
        
        This releases intermediate tensors and tracker history from VRAM
        without unloading the model weights.
        """
        import gc
        import torch
        
        # Force Python garbage collection first
        gc.collect()
        
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("MinimapDetector: CUDA cache cleared")

    def Cleanup(self) -> None:
        """Cleanup all CUDA resources (call before destroying detector)."""
        try:
            self.detect_engine_.ResetCudaState()
        except Exception as e:
            logger.debug(f"Cleanup detect_engine error: {e}")

        try:
            self.pose_engine_.ResetCudaState()
        except Exception as e:
            logger.debug(f"Cleanup pose_engine error: {e}")

    # -----------------------------------------------------------
    # Private helpers: YOLO parsing
    # -----------------------------------------------------------
    def _parseResults(self, yolo_results) -> List[Dict]:
        """Convert YOLO Results to list of dicts."""
        detections: List[Dict] = []

        for result in yolo_results:
            boxes = result.boxes
            if boxes is None or boxes.shape[0] == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                detections.append(
                    {
                        "cls": int(cls_ids[i]),
                        "confidence": float(conf[i]),
                        "bbox": tuple(xyxy[i].tolist()),
                    }
                )

        return detections

    def _findDetectionByClass(
        self, yolo_results: List, class_id: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Find the first detection of a given class."""
        for result in yolo_results:
            boxes = result.boxes
            if boxes is None or boxes.shape[0] == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                if int(cls_ids[i]) == class_id:
                    return xyxy, cls_ids, i

        return None

    def _findDetectionCenter(
        self, yolo_results: List, class_id: int
    ) -> Optional[Tuple[float, float]]:
        """Get center of first detection of given class."""
        found = self._findDetectionByClass(yolo_results, class_id)
        if found is None:
            return None

        xyxy, _, idx = found
        x1, y1, x2, y2 = xyxy[idx]
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def _findDetectionBbox(
        self, yolo_results: List, class_id: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """Get bbox of first detection of given class."""
        found = self._findDetectionByClass(yolo_results, class_id)
        if found is None:
            return None

        xyxy, _, idx = found
        x1, y1, x2, y2 = xyxy[idx]
        return (float(x1), float(y1), float(x2), float(y2))

    # -----------------------------------------------------------
    # Private: arrow crop and pose detection
    # -----------------------------------------------------------
    def _cropArrowRegion(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """Crop arrow region based on bbox."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        crop_x1 = int(max(0, x1))
        crop_y1 = int(max(0, y1))
        crop_x2 = int(min(w, x2))
        crop_y2 = int(min(h, y2))

        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            logger.debug("MinimapDetector: invalid crop region")
            return None, (0, 0)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            logger.debug("MinimapDetector: empty crop")
            return None, (0, 0)

        return crop, (crop_x1, crop_y1)

    def _detectArrowPose(
        self, crop: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """Detect arrow pose (center and angle) with pose model then fallback to OpenCV."""
        pose_result = self.pose_engine_.Detect(
            crop,
            confidence_threshold=self.conf_threshold_,
            iou_threshold=self.iou_threshold_,
            max_det=10,
        )

        pose_data = self._parsePoseResult(pose_result)
        if pose_data is not None:
            return pose_data

        local_center, raw_angle = self.arrow_detector_.detect(crop)
        return local_center, raw_angle

    def _parsePoseResult(
        self, pose_results: List
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """Parse pose model keypoints (head, L, R)."""
        if not pose_results or len(pose_results) == 0:
            return None

        result = pose_results[0]

        if not hasattr(result, "keypoints") or result.keypoints is None:
            return None

        keypoints = result.keypoints
        if keypoints.shape[0] == 0:
            return None

        kpts = keypoints.data[0]

        if kpts.shape[0] < 3:
            logger.debug("MinimapDetector: pose keypoints < 3 (need head, L, R)")
            return None

        if hasattr(kpts, "cpu"):
            kpts = kpts.cpu().numpy()
        else:
            kpts = np.array(kpts)

        head, L, R = kpts[0], kpts[1], kpts[2]
        head_conf = head[2] if len(head) > 2 else 1.0
        L_conf = L[2] if len(L) > 2 else 1.0
        R_conf = R[2] if len(R) > 2 else 1.0

        if (
            head_conf < self.min_keypoint_conf_
            or L_conf < self.min_keypoint_conf_
            or R_conf < self.min_keypoint_conf_
        ):
            logger.debug(
                "MinimapDetector: keypoint confidence too low "
                f"(head={head_conf:.2f}, L={L_conf:.2f}, R={R_conf:.2f})"
            )
            return None

        center_x = (head[0] + L[0] + R[0]) / 3.0
        center_y = (head[1] + L[1] + R[1]) / 3.0
        local_center = (float(center_x), float(center_y))

        bottom_center_x = (L[0] + R[0]) / 2.0
        bottom_center_y = (L[1] + R[1]) / 2.0
        dx = head[0] - bottom_center_x
        dy = head[1] - bottom_center_y
        angle_rad = np.arctan2(dy, dx)
        raw_angle = np.degrees(angle_rad) % 360.0

        return (local_center, float(raw_angle))

    # -----------------------------------------------------------
    # Private: failure frame capture
    # -----------------------------------------------------------
    def _maybe_save_failure(self, minimap_frame: Optional[np.ndarray], arrow_crop: Optional[np.ndarray], kind: str) -> None:
        """Save failure samples at a limited rate for training."""
        if not self.fail_capture_enabled_:
            return

        now = time.perf_counter()
        if now - self._last_fail_save_ts < self.fail_capture_interval_s_:
            return

        if minimap_frame is None and arrow_crop is None:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        try:
            if kind == "arrow" and arrow_crop is not None:
                out_path = self.fail_arrow_dir_ / f"{ts}.png"
                cv2.imwrite(str(out_path), arrow_crop)
            elif minimap_frame is not None:
                out_path = self.fail_minimap_dir_ / f"{ts}.png"
                cv2.imwrite(str(out_path), minimap_frame)
            else:
                return

            self._last_fail_save_ts = now
            logger.debug(f"Failure frame saved: {out_path}")
        except Exception as e:
            logger.error(f"Failed to save failure frame: {e}")
