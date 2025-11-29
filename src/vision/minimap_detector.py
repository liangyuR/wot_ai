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

from src.vision.detection_engine import DetectionEngine


@dataclass
class MinimapDetectionResult:
    """MinimapDetector 的统一输出结构"""
    self_pos: Optional[Tuple[float, float]] # center
    self_angle: Optional[float] # angle
    enemy_flag_pos: Optional[Tuple[float, float]] # center
    raw_detections: List[Dict] # [{"cls": int, "confidence": float, "bbox": Tuple[float, float, float, float]}]


class AngleSmoother:
    """角度平滑器：平滑角度变化，避免抖动"""
    
    def __init__(self, alpha: float = 0.25, max_step_deg: float = 45.0):
        """初始化角度平滑器
        
        Args:
            alpha: 平滑系数，0 < alpha <= 1，值越小越平滑但响应越慢
            max_step_deg: 单帧最大允许变化角度（度），超过此值会被限幅
        """
        self.alpha_ = max(0.0, min(1.0, alpha))
        self.max_step_deg_ = max(0.0, max_step_deg)
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
            return None
        
        if not self.valid_:
            self.angle_ = raw_angle
            self.valid_ = True
            return self.angle_
        
        # 计算角度差（处理跨越0°/360°边界的情况）
        diff = (raw_angle - self.angle_ + 540) % 360 - 180
        
        # 限幅：单帧最大变化角度
        if abs(diff) > self.max_step_deg_:
            diff = self.max_step_deg_ if diff > 0 else -self.max_step_deg_
        
        # === 自适应 alpha：抖动小就抑制，大幅转向就加快响应 ===
        abs_diff = abs(diff)
        # 你可以把这些阈值配到 config 里，我先写死一版
        if abs_diff < 2.0:
            # 极小变化，当成噪声，减弱更新
            effective_alpha = self.alpha_ * 0.4
        elif abs_diff < 10.0:
            # 正常缓慢转向，用原始 alpha
            effective_alpha = self.alpha_
        else:
            # 大幅转向，放大 alpha，加快跟踪
            effective_alpha = min(1.0, self.alpha_ * 2.0)

        self.angle_ = (self.angle_ + effective_alpha * diff) % 360.0
        
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
        model_path_base: str,
        model_path_arrow: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.25,
        smoothing_alpha: float = 0.25,
        max_step_deg: float = 45.0,
        min_area_ratio: float = 0.2,
        max_area_ratio: float = 0.9,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 3.0,
    ):
        if not model_path_base:
            raise ValueError("model_path 不能为空")
        if not model_path_arrow:
            raise ValueError("model_path_arrow 不能为空")

        # 基地检测模块
        self.base_engine_ = DetectionEngine(model_path_base)
        # 箭头检测模块
        self.arrow_engine_ = DetectionEngine(model_path_arrow)

        self.conf_threshold_ = conf_threshold
        self.iou_threshold_ = iou_threshold
        self.class_name_to_id_: Dict[str, int] = {}
        
        # 角度平滑器参数
        self.angle_smoother_ = AngleSmoother(
            alpha=smoothing_alpha,
            max_step_deg=max_step_deg,
        )
        
        # 轮廓过滤参数
        self.min_area_ratio_ = min_area_ratio
        self.max_area_ratio_ = max_area_ratio
        self.min_aspect_ratio_ = min_aspect_ratio
        self.max_aspect_ratio_ = max_aspect_ratio

        # 基地位置缓存，初始化为 None，只有成功检测到 enemy_flag 后才设置
        self.base_position: Optional[Tuple[float, float]] = None

    # -----------------------------------------------------------
    def LoadModel(self) -> bool:
        """加载模型并初始化类别名称映射"""
        if not self.base_engine_.LoadModel():
            logger.error("MinimapDetector: 模型加载失败")
            return False

        if not self.arrow_engine_.LoadModel():
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
        if frame is None or frame.size == 0:
            logger.error("MinimapDetector: frame 为空或无效")
            return MinimapDetectionResult(None, None, None, [])

        # 检测基地位置（enemy_flag）
        # 基地位置不会移动，检测到后可以缓存，后续帧跳过检测以提高性能
        if self.base_position is not None:
            # 已缓存基地位置，跳过检测
            raw_detections = []
        else:
            base_result = self.base_engine_.Detect(
                frame, 
                confidence_threshold=self.conf_threshold_, 
                iou_threshold=self.iou_threshold_,
                max_det=10  # base_engine 主要用于检测 enemy_flag
            )
            logger.debug(f"base_engine 检测结果: {base_result}")

            if not base_result:
                logger.error("MinimapDetector: base_engine 检测结果为空，无法获取 enemy_flag 位置")
                return MinimapDetectionResult(None, None, None, [])

            if not hasattr(self, "_class_mapping_initialized") or not self._class_mapping_initialized:
                self._update_class_mapping(base_result)
                self._class_mapping_initialized = True

            # 解析 Results 对象为字典列表
            raw_detections = self._parse_results(base_result)
            
            # 提取 enemy_flag 的中心位置并缓存
            self.base_position = self._extract_center(raw_detections, "enemy_flag")
            if self.base_position is not None:
                logger.debug(f"检测到 enemy_flag 位置并缓存: {self.base_position}")
            else:
                logger.warning("MinimapDetector: 未能在检测结果中找到 enemy_flag")

        # 检测箭头
        arrow_result = self.arrow_engine_.Detect(
            frame,
            confidence_threshold=self.conf_threshold_,
            iou_threshold=self.iou_threshold_,
            max_det=1,  # 只需要检测1个目标（self_arrow）
        )
        
        # arrow_result 是 Results 对象列表，需要先检查是否为空
        if not arrow_result or len(arrow_result) == 0:
            logger.warning("MinimapDetector: 箭头检测结果为空")
            return MinimapDetectionResult(None, None, None, raw_detections)
        
        # 获取第一个检测结果的 keypoints
        first_result = arrow_result[0]
        if not hasattr(first_result, "keypoints") or first_result.keypoints is None:
            logger.warning("MinimapDetector: 箭头检测结果中没有 keypoints")
            return MinimapDetectionResult(None, None, None, raw_detections)

        import os
        plotted = first_result.plot()  # 返回绘制好结果的 BGR 图像
        out_dir = "C:/Users/11601/project/wot_ai/resource/run/"
        os.makedirs(out_dir, exist_ok=True)
        # 按已存在文件数量递增命名
        base_name = "arrow"
        ext = ".png"
        i = 1
        while True:
            out_path = os.path.join(out_dir, f"{base_name}{i}{ext}")
            if not os.path.exists(out_path):
                break
            i += 1
        cv2.imwrite(out_path, plotted)
        
        keypoints = first_result.keypoints
        if keypoints.xy is None or len(keypoints.xy) == 0:
            logger.warning("MinimapDetector: keypoints 坐标为空")
            return MinimapDetectionResult(None, None, None, raw_detections)

        xy = keypoints.xy  # shape: (N, K, 2)，N为检测框数量，K为关键点数量

        # 验证 xy 数组的有效性
        if len(xy) == 0:
            logger.warning("MinimapDetector: keypoints.xy 数组为空")
            return MinimapDetectionResult(None, None, None, raw_detections)
        
        # 只取第一个检测结果
        coords = xy[0]  # shape: (K, 2)
        
        # 验证 coords 数组的有效性
        if coords is None or len(coords) == 0:
            logger.warning("MinimapDetector: coords 数组为空")
            return MinimapDetectionResult(None, None, None, raw_detections)
        
        # 解析箭头关键点：
        # - coords[0] 是 head（箭头头部中心）
        # - coords[1] 是 tail（箭头尾部中心）
        # - 角度从 tail 指向 head（尾部指向头部），表示箭头朝向
        # - 中心位置是 head 和 tail 的中点
        if coords.shape[0] >= 2:
            try:
                # 提取关键点坐标，确保转换为标量
                head = coords[0]  # 箭头头部中心，shape: (2,)
                tail = coords[1]  # 箭头尾部中心，shape: (2,)
                
                # 转换为 Python 标量，避免 numpy 类型问题
                head_x = float(head[0])
                head_y = float(head[1])
                tail_x = float(tail[0])
                tail_y = float(tail[1])
                
                # 验证坐标有效性（检查 NaN 和 inf）
                if not (np.isfinite(head_x) and np.isfinite(head_y) and 
                        np.isfinite(tail_x) and np.isfinite(tail_y)):
                    logger.warning("MinimapDetector: 关键点坐标包含 NaN 或 inf")
                    self_pos = None
                    self_angle = None
                else:
                    # 计算中心位置（head 和 tail 的中点）
                    self_pos = ((head_x + tail_x) / 2, (head_y + tail_y) / 2)
                    
                    # 计算角度：从 tail 指向 head 的方向
                    # dx = head_x - tail_x 表示从 tail 到 head 的 x 方向
                    # dy = head_y - tail_y 表示从 tail 到 head 的 y 方向
                    # 由于图像坐标系 y 轴向下，需要取反 dy 来转换为数学坐标系
                    dx = head_x - tail_x
                    dy = head_y - tail_y
                    
                    # 检查 dx 和 dy 是否都为 0（避免除零或无效角度）
                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        logger.warning("MinimapDetector: head 和 tail 位置相同，无法计算角度")
                        self_angle = None
                    else:
                        self_angle = float(np.degrees(np.arctan2(-dy, dx)) % 360)
            except (IndexError, TypeError, ValueError) as e:
                logger.error(f"MinimapDetector: 解析关键点时发生异常: {e}")
                self_pos = None
                self_angle = None
        else:
            logger.warning("MinimapDetector: keypoints 少于2个")
            self_pos = None
            self_angle = None

        # enemy_flag_pos 从缓存的基地位置获得
        enemy_flag_pos = self.base_position

        result = MinimapDetectionResult(
            self_pos=self_pos,
            self_angle=self_angle,
            enemy_flag_pos=enemy_flag_pos,
            raw_detections=raw_detections,
        )
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
        
        if len(approx) < 4:
            return None
        
        # 6. 轮廓质量过滤
        # 计算轮廓面积和 bbox 面积
        # contour_area = cv2.contourArea(largest_contour)
        # bbox_area = (x2_clipped - x1_clipped) * (y2_clipped - y1_clipped)
        # if bbox_area <= 0:
        #     return None
        
        # area_ratio = contour_area / bbox_area
        # if area_ratio < self.min_area_ratio_ or area_ratio > self.max_area_ratio_:
        #     return None
        
        # 检查外接矩形宽高比
        # x, y, w, h = cv2.boundingRect(largest_contour)
        # if h <= 0:
        #     return None
        # aspect_ratio = w / h
        # if aspect_ratio < self.min_aspect_ratio_ or aspect_ratio > self.max_aspect_ratio_:
        #     return None
        
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
        self.base_position = None

def run_benchmark(
    image_path: str,
    model_path_base: str,
    model_path_arrow: str,
    target_fps: float,
    max_frames: Optional[int] = None,
    show_visualization: bool = False,
) -> None:
    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        sys.exit(1)

    # 创建检测器
    detector = MinimapDetector(model_path_base=model_path_base, model_path_arrow=model_path_arrow)
    if not detector.LoadModel():
        print("模型加载失败")
        sys.exit(1)

    # 预热一次
    _ = detector.Detect(frame)

    interval = 1.0 / target_fps if target_fps > 0 else 0.0
    print(f"开始性能测试: target_fps={target_fps}, interval={interval*1000:.2f} ms, max_frames={max_frames}")

    times_ms = []
    frame_count = 0
    start_all = time.time()

    try:
        while True:
            t0 = time.time()
            result = detector.Detect(frame)
            t1 = time.time()

            infer_ms = (t1 - t0) * 1000.0
            times_ms.append(infer_ms)
            frame_count += 1

            if show_visualization:
                vis_img = detector._visualize_results(frame, result)
                cv2.imshow("MinimapDetector Vis", vis_img)
                # 非阻塞刷新一下窗口
                cv2.waitKey(1)

            # 维持目标 FPS（简单 sleep）
            if interval > 0:
                elapsed = t1 - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if max_frames is not None and frame_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n收到中断信号，结束测试……")

    end_all = time.time()

    if show_visualization:
        cv2.destroyAllWindows()

    if not times_ms:
        print("没有成功运行任何帧")
        return

    total_time = end_all - start_all
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    effective_fps = frame_count / total_time if total_time > 0 else 0.0

    print("\n===== MinimapDetector 性能统计 =====")
    print(f"总帧数: {frame_count}")
    print(f"总耗时: {total_time*1000:.2f} ms")
    print(f"平均推理耗时: {avg_ms:.2f} ms")
    print(f"最小推理耗时: {min_ms:.2f} ms")
    print(f"最大推理耗时: {max_ms:.2f} ms")
    print(f"目标 FPS: {target_fps}")
    print(f"实际 FPS: {effective_fps:.2f}")


if __name__ == "__main__":
    import argparse
    import time
    import sys
    parser = argparse.ArgumentParser(
        description="测试 MinimapDetector 检测性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="图片文件路径（小地图截图）",
    )
    parser.add_argument(
        "--model_path_base",
        type=str,
        default="best.pt",
        help="检测模型路径 (默认: best.pt)",
    )
    parser.add_argument(
        "--model_path_arrow",
        type=str,
        default="best.pt",
        help="检测模型路径 (默认: best.pt)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="目标测试 FPS，决定循环节奏 (默认: 10)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="最大测试帧数，None 表示无限循环直到 Ctrl+C (默认: 200)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否实时显示可视化结果（会稍微影响性能）",
    )

    args = parser.parse_args()

    max_frames = args.frames if args.frames > 0 else None

    run_benchmark(
        image_path=args.image_path,
        model_path_base=args.model_path_base,
        model_path_arrow=args.model_path_arrow,
        target_fps=args.fps,
        max_frames=max_frames,
        show_visualization=True,
    )
