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

import numpy as np
from loguru import logger
import cv2

from .detection_engine import DetectionEngine


@dataclass
class MinimapDetectionResult:
    """MinimapDetector 的统一输出结构"""
    self_pos: Optional[Tuple[float, float]] # center
    self_angle: Optional[float] # angle
    enemy_flag_pos: Optional[Tuple[float, float]] # center
    raw_detections: List[Dict] # [{"cls": int, "confidence": float, "bbox": Tuple[float, float, float, float]}]


class AngleSmoother:
    """角度平滑器：平滑角度变化，避免抖动"""
    
    def __init__(self):
        self.angle_: Optional[float] = None
        self.valid_: bool = False
    
    def Update(self, raw_angle: Optional[float]) -> Optional[float]:
        """更新角度，返回平滑后的角度
        
        Args:
            raw_angle: 原始角度（度），如果为None则返回当前角度
        
        Returns:
            平滑后的角度（度）或 None
        """
        if raw_angle is None:
            return self.angle_ if self.valid_ else None
        
        if not self.valid_:
            self.angle_ = raw_angle
            self.valid_ = True
            return self.angle_
        
        # 计算角度差（处理跨越0°/360°边界的情况）
        diff = (raw_angle - self.angle_ + 540) % 360 - 180
        self.angle_ = (self.angle_ + diff) % 360.0
        
        return self.angle_
    
    def Reset(self):
        """重置平滑器"""
        self.angle_ = None
        self.valid_ = False


class MinimapDetector:
    """
    重新设计后的小地图检测器

    输入：一张完整的小地图图像 (frame_minimap)
    输出：self_pos, enemy_flag_pos 以及原始 YOLO detection 列表
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.25,
    ):
        if not model_path:
            raise ValueError("model_path 不能为空")

        self.engine_ = DetectionEngine(model_path)
        self.conf_threshold_ = conf_threshold
        self.iou_threshold_ = iou_threshold
        self.class_name_to_id_: Dict[str, int] = {}
        self.angle_smoother_ = AngleSmoother()  # 角度平滑器

    # -----------------------------------------------------------
    def LoadModel(self) -> bool:
        """加载模型并初始化类别名称映射"""
        if not self.engine_.LoadModel():
            logger.error("MinimapDetector: 模型加载失败")
            return False

        logger.info("MinimapDetector: 模型加载成功")
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
    def Detect(self, frame: np.ndarray) -> MinimapDetectionResult:
        """
        检测小地图中的关键元素

        Args:
            frame: 小地图 BGR 图像
            debug: 是否显示调试可视化

        Returns:
            MinimapDetectionResult
        """
        import time
        total_start = time.time()
        
        if frame is None or frame.size == 0:
            logger.error("MinimapDetector: frame 为空或无效")
            return MinimapDetectionResult(None, None, None, [])

        # 调用检测引擎，返回 YOLO Results 对象（列表）
        yolo_start = time.time()
        yolo_results = self.engine_.Detect(
            frame, 
            confidence_threshold=self.conf_threshold_, 
            iou_threshold=self.iou_threshold_,
            max_det=10  # 优化：只需要检测2个目标（self_arrow和enemy_flag）
        )
        yolo_elapsed = time.time() - yolo_start
        
        if not yolo_results:
            logger.error("MinimapDetector: 检测结果为空")
            return MinimapDetectionResult(None, None, None, [])

        if not hasattr(self, "_class_mapping_initialized") or not self._class_mapping_initialized:
            self._update_class_mapping(yolo_results)
            self._class_mapping_initialized = True

        # 解析 Results 对象为字典列表
        parse_start = time.time()
        raw_detections = self._parse_results(yolo_results)
        parse_elapsed = time.time() - parse_start

        # 位置提取（使用类别名称）
        pos_start = time.time()
        self_pos = self._extract_center(raw_detections, "self_arrow")
        enemy_flag_pos = self._extract_center(raw_detections, "enemy_flag")
        pos_elapsed = time.time() - pos_start
        
        # 角度提取
        angle_start = time.time()
        self_angle = self._extract_angle(frame, raw_detections, "self_arrow")
        angle_elapsed = time.time() - angle_start

        result = MinimapDetectionResult(self_pos, self_angle, enemy_flag_pos, raw_detections)
        
        # 性能监控：记录各步骤耗时
        total_elapsed = time.time() - total_start
        if yolo_elapsed > 0.05:  # YOLO超过50ms
            logger.warning(f"[性能] YOLO推理耗时: {yolo_elapsed*1000:.1f}ms")
        if angle_elapsed > 0.01:  # 角度提取超过10ms
            logger.debug(f"[性能] 角度提取耗时: {angle_elapsed*1000:.1f}ms")
        if total_elapsed > 0.1:  # 总耗时超过100ms
            logger.warning(f"[性能] 检测总耗时: {total_elapsed*1000:.1f}ms (YOLO:{yolo_elapsed*1000:.1f}ms, 解析:{parse_elapsed*1000:.1f}ms, 位置:{pos_elapsed*1000:.1f}ms, 角度:{angle_elapsed*1000:.1f}ms)")

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
    def _extract_angle_geometric(
        self,
        detections: List[Dict],
        frame_bgr: np.ndarray,
    ) -> Optional[float]:
        """使用几何方法提取箭头角度。

        Args:
            detections: YOLO检测结果字典列表
            frame_bgr: 完整帧的BGR图像

        Returns:
            角度（度），失败返回None
        """
        if self.orientation_estimator_ is None:
            return None

        class_id = self._get_class_id_by_name("self_arrow")
        if class_id is None:
            return None

        # 找到 self_arrow 的检测结果（取置信度最高的）
        target_det = None
        best_conf = -1.0
        for det in detections:
            if int(det.get("cls", -1)) == int(class_id):
                conf = det.get("confidence", 0.0)
                if conf > best_conf:
                    best_conf = conf
                    target_det = det

        if target_det is None:
            return None

        bbox = target_det.get("bbox")
        if bbox is None or len(bbox) != 4:
            return None

        # 调用几何朝向估计器
        return self.orientation_estimator_.estimate_from_bbox(frame_bgr, bbox)

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

    def _extract_angle(self, minimap: np.ndarray, detections: List[Dict], class_name: str) -> Optional[float]:
        """从检测结果中提取角度（通过轮廓分析）
        
        Args:
            minimap: 小地图 BGR 图像
            detections: 检测结果字典列表
            class_name: 类别名称（如 "self_arrow"）
            debug: 是否显示调试信息
        
        Returns:
            角度（度）或 None
        """
        if minimap is None or minimap.size == 0:
            return None
        
        # 1. 找到 self_arrow 的 bbox
        class_id = self._get_class_id_by_name(class_name)
        if class_id is None:
            return None
        
        filtered = [d for d in detections if d["cls"] == class_id]
        if not filtered:
            return None
        
        best = max(filtered, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        
        # 2. 裁剪获取该区域
        h, w = minimap.shape[:2]
        x1_clipped = max(0, min(x1, w - 1))
        y1_clipped = max(0, min(y1, h - 1))
        x2_clipped = max(x1_clipped + 1, min(x2, w))
        y2_clipped = max(y1_clipped + 1, min(y2, h))
        
        roi = minimap[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
        if roi.size == 0:
            return None
        
        # 3. 灰度 + OTSU 二值化（简化版：箭头是白色，直接使用OTSU）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. 找最大轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        # 5. approxPolyDP 得到 4 顶点
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) != 4:
            return None
        
        # 返回 4 个顶点坐标（用于后续计算角度）
        vertices = approx.reshape(-1, 2)
        raw_angle = self.compute_angle_from_vertices(vertices)
        
        # 使用角度平滑器平滑角度
        smoothed_angle = self.angle_smoother_.Update(raw_angle)
        return smoothed_angle

    def compute_angle_from_vertices(self, vertices):
        """
        输入：vertices shape = (4, 2)
        输出：角度 (0°=向右, 90°=向上) 或 None
        """
        if vertices is None or len(vertices) != 4:
            return None

        pts = np.array(vertices, dtype=np.float32)  # shape (4,2)

        # 工具：三点之间的 pairwise 距离
        def pairwise_dist(idx_list):
            dists = []
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    p = pts[idx_list[i]]
                    q = pts[idx_list[j]]
                    dists.append(float(np.linalg.norm(p - q)))
            return dists

        best_tip_idx = None
        best_cluster_spread = None
        best_tail_center = None

        # 遍历四个点，尝试把每个点视为“远点 tip”
        for tip_idx in range(4):
            tail_indices = [i for i in range(4) if i != tip_idx]

            dists = pairwise_dist(tail_indices)
            if not dists:
                continue

            cluster_spread = max(dists)  # 三个近点之间最大距离
            if cluster_spread < 1e-6:
                continue

            tail_pts = pts[tail_indices]
            tail_center = tail_pts.mean(axis=0)

            tip_pt = pts[tip_idx]
            tip_dist = float(np.linalg.norm(tip_pt - tail_center))

            ratio = tip_dist / cluster_spread

            # tip 必须比尾部尺寸明显更远
            if ratio < 1.2:  # 经验阈值，可调
                continue

            # 挑尾部最紧凑的（cluster_spread 最小）
            if best_tip_idx is None or cluster_spread < best_cluster_spread:
                best_tip_idx = tip_idx
                best_cluster_spread = cluster_spread
                best_tail_center = tail_center

        # 如果没有找到合理的 tip，就直接返回 None
        if best_tip_idx is None:
            return None

        tip_x, tip_y = pts[best_tip_idx]
        tail_cx, tail_cy = best_tail_center

        dx = tip_x - tail_cx
        dy = tail_cy - tip_y     
        if dx * dx + dy * dy < 1e-6:
            return None

        # 图像坐标转数学坐标 (y 取反)
        angle_rad = np.arctan2(-dy, dx)
        angle_deg = float(np.degrees(angle_rad))
        return angle_deg

    # -----------------------------------------------------------
    def Reset(self):
        """重置检测器状态"""
        self.angle_smoother_.Reset()

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description="测试 MinimapDetector 检测功能",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="图片文件路径（小地图截图）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best.pt",
        help="检测模型路径 (默认: best.pt)"
    )
    args = parser.parse_args()

    image_path = args.image_path
    model_path = args.model_path
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        sys.exit(1)

    detector = MinimapDetector(model_path=model_path)
    if not detector.LoadModel():
        print("模型加载失败")
        sys.exit(1)
    result = detector.Detect(frame)
    print("检测结果:", result)

    vis_img = detector._visualize_results(frame, result)
    cv2.imshow("MinimapDetector Vis", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()