#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重写后的 MinimapDetector：
- 使用全新的 DetectionEngine
- 使用结构化 Detection
- 统一 class ID 常量
- 返回结构更清晰（位置 + 原始检测）
- 更易扩展后续的“朝向 + 障碍物 + 其他元素”
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import time

import numpy as np
from loguru import logger
import cv2

from ..detection_engine import DetectionEngine


@dataclass
class MinimapDetectionResult:
    """MinimapDetector 的统一输出结构"""
    self_pos: Optional[Tuple[float, float]] # center
    self_angle: Optional[float] # angle
    enemy_flag_pos: Optional[Tuple[float, float]] # center
    raw_detections: List[Dict] # [{"cls": int, "confidence": float, "bbox": Tuple[float, float, float, float]}]


class MinimapDetector:
    """
    重新设计后的小地图检测器

    输入：一张完整的小地图图像 (frame_minimap)
    输出：self_pos, enemy_flag_pos 以及原始 YOLO detection 列表
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.75,
        iou_threshold: float = 0.75,
        input_size: Optional[Tuple[int, int]] = None,
        preprocess_input: bool = False,
    ):
        if not model_path:
            raise ValueError("model_path 不能为空")

        self.engine_ = DetectionEngine(model_path)
        self.conf_threshold_ = conf_threshold
        self.iou_threshold_ = iou_threshold
        self.input_size_ = input_size  # 如果指定，会将输入resize到此尺寸
        self.preprocess_input_ = preprocess_input  # 是否启用预处理（resize）
        self.class_name_to_id_: Dict[str, int] = {}

    # -----------------------------------------------------------
    def LoadModel(self) -> bool:
        """加载模型并初始化类别名称映射"""
        if not self.engine_.LoadModel():
            return False

        return True
    
    def _update_class_mapping(self, results) -> None:
        """从 Results 对象更新类别名称映射"""
        if not results:
            return
        
        # results 是列表，取第一个
        result = results[0] if isinstance(results, list) else results
        if hasattr(result, "names") and isinstance(result.names, dict):
            # names 是 {id: name} 的字典，反转成 {name: id}
            self.class_name_to_id_.clear()
            for class_id, class_name in result.names.items():
                self.class_name_to_id_[str(class_name)] = int(class_id)
            logger.debug(f"类别映射已更新: {self.class_name_to_id_}")
    
    def _get_class_id_by_name(self, class_name: str) -> Optional[int]:
        """通过类别名称获取类别 ID
        
        Args:
            class_name: 类别名称（如 "self_arrow", "enemy_flag"）
        
        Returns:
            类别 ID，如果未找到则返回 None
        """
        return self.class_name_to_id_.get(class_name)

    # -----------------------------------------------------------
    def Detect(self, frame: np.ndarray, debug: bool = False) -> MinimapDetectionResult:
        """
        检测小地图中的关键元素

        Args:
            frame: 小地图 BGR 图像
            debug: 是否显示调试可视化

        Returns:
            MinimapDetectionResult
        """
        if frame is None or frame.size == 0:
            logger.error("MinimapDetector: frame 为空或无效")
            return MinimapDetectionResult(None, None, None, [])

        # 诊断：记录输入图像尺寸（首次或异常时）
        if not hasattr(self, '_last_frame_shape') or self._last_frame_shape != frame.shape:
            h, w = frame.shape[:2]
            logger.debug(f"MinimapDetector输入尺寸: {w}x{h}")
            if w > 800 or h > 800:
                logger.warning(f"小地图尺寸较大 ({w}x{h})，如果性能慢可考虑resize")
            self._last_frame_shape = frame.shape

        # 可选预处理：resize到固定尺寸（如果启用）
        if self.preprocess_input_ and self.input_size_ is not None:
            target_w, target_h = self.input_size_
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 调用检测引擎，返回 YOLO Results 对象（列表）
        yolo_results = self.engine_.Detect(
            frame, 
            confidence_threshold=self.conf_threshold_, 
            iou_threshold=self.iou_threshold_
        )
        
        if not yolo_results:
            return MinimapDetectionResult(None, None, None, [])

        if not hasattr(self, "_class_mapping_initialized") or not self._class_mapping_initialized:
            self._update_class_mapping(yolo_results)
            self._class_mapping_initialized = True

        # 解析 Results 对象为字典列表
        raw_detections = self._parse_results(yolo_results)

        # 位置提取（使用类别名称）
        self_pos = self._extract_center(raw_detections, "self_arrow")
        enemy_flag_pos = self._extract_center(raw_detections, "enemy_flag")
        
        # 角度提取已移除，self_angle 始终为 None
        self_angle = None

        result = MinimapDetectionResult(self_pos, self_angle, enemy_flag_pos, raw_detections)
        
        if debug:
            vis_img = self._visualize_results(frame, result)
            if vis_img is not None:
                cv2.imshow("MinimapDetector Debug", vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return result

    def _parse_results(self, yolo_results) -> List[Dict]:
        """解析 YOLO Results 对象为字典列表
        
        Args:
            yolo_results: YOLO Results 对象（列表）
        
        Returns:
            检测结果字典列表
        """
        detections: List[Dict] = []
        
        # yolo_results 是列表，每个元素是一个 Results 对象
        for result in yolo_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                det_dict = {
                    "cls": cls_id,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                }
                
                # 如果有 mask，也添加
                if result.masks is not None and hasattr(result.masks, "data") and i < result.masks.data.shape[0]:
                    try:
                        mask_tensor = result.masks.data[i]
                        mask_np = mask_tensor.cpu().numpy().astype(np.float32)
                        # 调整到原始尺寸
                        orig_h, orig_w = result.orig_shape
                        if mask_np.shape != (orig_h, orig_w):
                            mask_np = cv2.resize(
                                mask_np,
                                (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST
                            )
                        mask_np = (mask_np > 0.5).astype(np.uint8)
                        det_dict["mask"] = mask_np
                    except Exception as e:
                        logger.warning(f"提取 mask 失败（索引 {i}）: {e}")
                
                detections.append(det_dict)
        
        return detections

    def _visualize_results(self, frame_minimap: np.ndarray, results: MinimapDetectionResult) -> np.ndarray:
        """可视化检测结果
        
        Args:
            frame_minimap: 小地图 BGR 图像
            results: MinimapDetectionResult 对象
        
        Returns:
            绘制了检测结果的可视化图像
        """
        vis_img = frame_minimap.copy()
        
        # 绘制所有检测的 mask，而不是检测框
        for det in results.raw_detections:
            cls_id = det["cls"]
            color = (0, 255, 0) if cls_id == self._get_class_id_by_name("self_arrow") else (0, 0, 255)
            if "mask" in det:
                mask = det["mask"]  # uint8, 单通道
                # 生成彩色 mask 叠加
                color_mask = np.zeros_like(vis_img, dtype=np.uint8)
                color_mask[mask.astype(bool)] = color
                # 使用 addWeighted 合成效果
                vis_img = cv2.addWeighted(vis_img, 1.0, color_mask, 0.5, 0)
            else:
                # 没有 mask 时可以选择降级为画框
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_id}:{det['confidence']:.2f}"
            # 选择一个合适的文本显示点
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.putText(vis_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制 self_arrow 的调试信息
        if results.self_pos:
            cx, cy = map(int, results.self_pos)
            label = "Self"
            if results.self_angle is not None:
                label += f" {results.self_angle:.1f}°"
                logger.info(f"[DEBUG] 检测到的角度: {results.self_angle:.2f}°")
            cv2.putText(vis_img, label, (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 绘制箭头方向
            if results.self_angle is not None:
                angle = results.self_angle
                angle_rad = np.radians(angle)
                arrow_length = 40
                end_x = int(cx + arrow_length * np.cos(angle_rad))
                end_y = int(cy + arrow_length * np.sin(angle_rad))
                cv2.arrowedLine(vis_img, (cx, cy), (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)

        if results.enemy_flag_pos:
            ex, ey = map(int, results.enemy_flag_pos)
            cv2.rectangle(vis_img, (ex-10, ey-10), (ex+10, ey+10), (0, 0, 255), 2)
            cv2.putText(vis_img, "Enemy", (ex+12, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return vis_img
        
    # -----------------------------------------------------------
    
    def _extract_center(self, detections: List[Dict], class_name: str) -> Optional[Tuple[float, float]]:
        """从检测列表中筛选某个类别，并取最高置信度的中心点
        
        Args:
            detections: 检测结果字典列表
            class_name: 类别名称（如 "self_arrow", "enemy_flag"）
        
        Returns:
            中心点坐标 (x, y) 或 None
        """
        class_id = self._get_class_id_by_name(class_name)
        if class_id is None:
            return None
        
        filtered = [d for d in detections if d["cls"] == class_id]
        if not filtered:
            return None

        best = max(filtered, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return (center_x, center_y)

    # -----------------------------------------------------------
    def Reset(self):
        pass