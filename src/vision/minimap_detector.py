#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinimapDetector：小地图检测器

检测流程：
1. 使用 detect_engine_（检测模型）在全图上检测 self_arrow 和 enemy_flag
2. 如果找到 self_arrow bbox，裁剪后优先使用 pose_engine_（姿态模型）检测关键点
3. 通过关键点计算精确的位置和朝向（Head=0, Tail=1）
4. 如果 pose_engine_ 检测失败，回退到 OpencvArrowDetector 检测中心和角度

输入：一张完整的小地图图像 (frame_minimap)
输出：self_pos, self_angle, enemy_flag_pos 以及原始检测结果列表
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
from loguru import logger
import cv2

from src.vision.detection_engine import DetectionEngine
from src.vision.opencv_arrow_detector import OpencvArrowDetector
from src.utils.angle_smoother import AngleSmoother
from src.utils.global_path import GetGlobalConfig


@dataclass
class MinimapDetectionResult:
    """MinimapDetector 的统一输出结构"""
    self_pos: Optional[Tuple[float, float]]        # 自身位置 (x, y)
    self_angle: Optional[float]                     # 自身朝向角度
    enemy_flag_pos: Optional[Tuple[float, float]]  # 敌方基地位置
    raw_detections: List[Dict]                      # 原始检测结果


class MinimapDetector:
    """双模型小地图检测器

    检测流程：
    1. 使用 detect_engine_（检测模型）在全图上检测 self_arrow 和 enemy_flag
    2. 如果找到 self_arrow bbox，裁剪后优先使用 pose_engine_（姿态模型）检测关键点
    3. 通过关键点计算精确的位置和朝向（Head=0, Tail=1）
    4. 如果 pose_engine_ 检测失败，回退到 OpencvArrowDetector 检测中心和角度
    """

    def __init__(self):
        config = GetGlobalConfig()

        # 检测参数
        self.conf_threshold_ = config.model.conf_threshold
        self.iou_threshold_ = config.model.iou_threshold
        self.crop_size_ = getattr(config.model, 'crop_size', 36)
        self.min_keypoint_conf_ = getattr(config.model, 'min_keypoint_conf', 0.5)

        # 检测引擎
        self.detect_engine_ = DetectionEngine(config.model.base_path)
        self.pose_engine_ = DetectionEngine(config.model.arrow_path)
        self.arrow_detector_ = OpencvArrowDetector()

        # 角度平滑器
        angle_cfg = config.angle_detection
        self.angle_smoother_ = AngleSmoother(
            alpha=angle_cfg.smoothing_alpha,
            max_step_deg=angle_cfg.max_step_deg,
            noise_threshold_deg=angle_cfg.noise_threshold_deg,
            normal_threshold_deg=angle_cfg.normal_threshold_deg,
            noise_alpha_factor=angle_cfg.noise_alpha_factor,
            large_turn_alpha_factor=angle_cfg.large_turn_alpha_factor,
        )

        # 基地位置缓存
        self.base_position_: Optional[Tuple[float, float]] = None

        # 类别 ID
        self.arrow_class_id_: int = config.model.class_id_arrow
        self.flag_class_id_: int = config.model.class_id_flag

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    def LoadModel(self) -> bool:
        """加载模型"""
        if not self.detect_engine_.LoadModel():
            logger.error("MinimapDetector: detect_engine 模型加载失败")
            return False

        if not self.pose_engine_.LoadModel():
            logger.error("MinimapDetector: pose_engine 模型加载失败")
            return False

        logger.info("MinimapDetector: 模型加载成功")
        return True

    def Detect(self, frame: np.ndarray) -> MinimapDetectionResult:
        """检测小地图中的目标

        Args:
            frame: 小地图 BGR 图像

        Returns:
            MinimapDetectionResult 包含位置、角度和原始检测结果
        """
        if frame is None or frame.size == 0:
            logger.debug("MinimapDetector: frame 为空或无效")
            return MinimapDetectionResult(None, None, None, [])

        # === 第一步：全局检测 ===
        detect_result = self.detect_engine_.Detect(
            frame,
            confidence_threshold=self.conf_threshold_,
            iou_threshold=self.iou_threshold_,
            max_det=10
        )

        if not detect_result or len(detect_result) == 0:
            logger.debug("MinimapDetector: detect_engine 检测结果为空")
            return MinimapDetectionResult(None, None, self.base_position_, [])

        # 解析检测结果
        raw_detections = self._parseResults(detect_result)

        # 检测并缓存 enemy_flag 位置（只需一次）
        if self.base_position_ is None:
            self.base_position_ = self._findDetectionCenter(
                detect_result, self.flag_class_id_
            )
            if self.base_position_ is not None:
                logger.info(f"检测到 enemy_flag 位置并缓存: {self.base_position_}")

        # 查找 self_arrow 的 bbox
        arrow_bbox = self._findDetectionBbox(detect_result, self.arrow_class_id_)
        if arrow_bbox is None:
            logger.debug("MinimapDetector: 未检测到 self_arrow")
            return MinimapDetectionResult(None, None, self.base_position_, raw_detections)

        # === 第二步：裁剪箭头区域 ===
        crop, crop_offset = self._cropArrowRegion(frame, arrow_bbox)
        if crop is None:
            return MinimapDetectionResult(None, None, self.base_position_, raw_detections)

        # === 第三步：检测箭头姿态 ===
        local_center, raw_angle = self._detectArrowPose(crop)

        # 转换为全局坐标
        self_pos = None
        if local_center is not None:
            cx_local, cy_local = local_center
            self_pos = (cx_local + crop_offset[0], cy_local + crop_offset[1])

        # 应用角度平滑
        self_angle = self.angle_smoother_.Update(raw_angle)

        return MinimapDetectionResult(
            self_pos=self_pos,
            self_angle=self_angle,
            enemy_flag_pos=self.base_position_,
            raw_detections=raw_detections,
        )

    def Reset(self) -> None:
        """重置检测器状态"""
        self.angle_smoother_.Reset()
        self.base_position_ = None

    # -----------------------------------------------------------
    # Private: YOLO 结果解析
    # -----------------------------------------------------------
    def _parseResults(self, yolo_results) -> List[Dict]:
        """解析 YOLO Results 对象为字典列表"""
        detections: List[Dict] = []

        for result in yolo_results:
            boxes = result.boxes
            if boxes is None or boxes.shape[0] == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                detections.append({
                    "cls": int(cls_ids[i]),
                    "confidence": float(conf[i]),
                    "bbox": tuple(xyxy[i].tolist()),
                })

        return detections

    def _findDetectionByClass(
        self,
        yolo_results: List,
        class_id: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """从 YOLO Results 中查找指定类别的第一个检测

        Args:
            yolo_results: YOLO Results 对象列表
            class_id: 类别 ID

        Returns:
            (xyxy_array, cls_array, index) 或 None
        """
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
        self,
        yolo_results: List,
        class_id: int
    ) -> Optional[Tuple[float, float]]:
        """从 YOLO Results 中提取指定类别的第一个检测框的中心点"""
        found = self._findDetectionByClass(yolo_results, class_id)
        if found is None:
            return None

        xyxy, _, idx = found
        x1, y1, x2, y2 = xyxy[idx]
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def _findDetectionBbox(
        self,
        yolo_results: List,
        class_id: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """从 YOLO Results 中提取指定类别的第一个检测框"""
        found = self._findDetectionByClass(yolo_results, class_id)
        if found is None:
            return None

        xyxy, _, idx = found
        x1, y1, x2, y2 = xyxy[idx]
        return (float(x1), float(y1), float(x2), float(y2))

    # -----------------------------------------------------------
    # Private: 箭头裁剪与姿态检测
    # -----------------------------------------------------------
    def _cropArrowRegion(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """裁剪箭头区域

        Args:
            frame: 原始图像
            bbox: 边界框 (x1, y1, x2, y2)

        Returns:
            (crop_image, (offset_x, offset_y)) 或 (None, (0, 0))
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        half_size = self.crop_size_ // 2
        h, w = frame.shape[:2]

        crop_x1 = int(max(0, cx - half_size))
        crop_y1 = int(max(0, cy - half_size))
        crop_x2 = int(min(w, cx + half_size))
        crop_y2 = int(min(h, cy + half_size))

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            logger.debug("MinimapDetector: 裁剪区域为空")
            return None, (0, 0)

        return crop, (crop_x1, crop_y1)

    def _detectArrowPose(
        self,
        crop: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """检测箭头姿态（中心和角度）

        优先使用 pose_engine_，失败则回退到 OpenCV 检测

        Args:
            crop: 裁剪后的箭头图像

        Returns:
            (local_center, raw_angle) 或 (None, None)
        """
        # 优先使用 pose_engine_
        pose_result = self.pose_engine_.Detect(
            crop,
            confidence_threshold=self.conf_threshold_,
            iou_threshold=self.iou_threshold_,
            max_det=10
        )

        pose_data = self._parsePoseResult(pose_result)
        if pose_data is not None:
            return pose_data

        # 回退：使用 OpenCV 箭头检测
        local_center, raw_angle = self.arrow_detector_.detect(crop)
        return local_center, raw_angle

    def _parsePoseResult(
        self,
        pose_results: List
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """解析 Pose 模型的关键点结果

        Args:
            pose_results: YOLO pose Results 对象列表

        Returns:
            (local_center, raw_angle) 或 None
        """
        if not pose_results or len(pose_results) == 0:
            return None

        result = pose_results[0]

        if not hasattr(result, "keypoints") or result.keypoints is None:
            return None

        keypoints = result.keypoints
        if keypoints.shape[0] == 0:
            return None

        kpts = keypoints.data[0]

        if kpts.shape[0] < 2:
            logger.debug("MinimapDetector: pose 模型关键点数量不足")
            return None

        if hasattr(kpts, "cpu"):
            kpts = kpts.cpu().numpy()
        else:
            kpts = np.array(kpts)

        # Head (index 0) 和 Tail (index 1)
        head = kpts[0]
        tail = kpts[1]

        head_conf = head[2] if len(head) > 2 else 1.0
        tail_conf = tail[2] if len(tail) > 2 else 1.0

        if head_conf < self.min_keypoint_conf_ or tail_conf < self.min_keypoint_conf_:
            logger.debug(
                f"MinimapDetector: 关键点置信度不足 "
                f"(head={head_conf:.2f}, tail={tail_conf:.2f})"
            )
            return None

        # 中心点
        center_x = (head[0] + tail[0]) / 2.0
        center_y = (head[1] + tail[1]) / 2.0
        local_center = (float(center_x), float(center_y))

        # 角度：Tail -> Head 方向
        dx = head[0] - tail[0]
        dy = head[1] - tail[1]
        angle_rad = np.arctan2(dy, dx)
        raw_angle = np.degrees(angle_rad) % 360.0

        return (local_center, float(raw_angle))