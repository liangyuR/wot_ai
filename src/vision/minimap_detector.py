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
from src.vision.opencv_arrow_detector import OpencvArrowDetector
from src.utils.global_path import GetGlobalConfig

@dataclass
class MinimapDetectionResult:
    """MinimapDetector 的统一输出结构"""
    self_pos: Optional[Tuple[float, float]] # center
    self_angle: Optional[float] # angle
    enemy_flag_pos: Optional[Tuple[float, float]] # center
    raw_detections: List[Dict] # [{"cls": int, "confidence": float, "bbox": Tuple[float, float, float, float]}]


class AngleSmoother:
    """角度平滑器：平滑角度变化，避免抖动
    
    特性：
    - 处理角度跨越 0°/360° 边界的情况
    - 自适应平滑系数：根据角度变化幅度动态调整
    - 输入验证：检查 NaN、inf 和角度范围
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        max_step_deg: float = 45.0,
        noise_threshold_deg: float = 2.0,
        normal_threshold_deg: float = 10.0,
        noise_alpha_factor: float = 0.4,
        large_turn_alpha_factor: float = 2.0,
    ):
        """初始化角度平滑器
        
        Args:
            alpha: 平滑系数，0 < alpha <= 1，值越小越平滑但响应越慢
            max_step_deg: 单帧最大允许变化角度（度），超过此值会被限幅
            noise_threshold_deg: 噪声阈值（度），小于此值视为噪声
            normal_threshold_deg: 正常转向阈值（度），小于此值视为正常转向
            noise_alpha_factor: 噪声时的 alpha 缩放因子
            large_turn_alpha_factor: 大幅转向时的 alpha 缩放因子
        """
        self.alpha_ = max(0.0, min(1.0, alpha))
        self.max_step_deg_ = max(0.0, max_step_deg)
        self.noise_threshold_deg_ = max(0.0, noise_threshold_deg)
        self.normal_threshold_deg_ = max(self.noise_threshold_deg_, normal_threshold_deg)
        self.noise_alpha_factor_ = max(0.0, noise_alpha_factor)
        self.large_turn_alpha_factor_ = max(1.0, large_turn_alpha_factor)
        
        self.angle_: Optional[float] = None
        self.valid_: bool = False
    
    def _normalize_angle(self, angle: float) -> float:
        """将角度归一化到 [0, 360) 范围"""
        return angle % 360.0
    
    def _compute_angle_diff(self, new_angle: float, old_angle: float) -> float:
        """计算角度差，处理跨越 0°/360° 边界的情况
        
        Returns:
            角度差（度），范围 [-180, 180]
        """
        return (new_angle - old_angle + 540.0) % 360.0 - 180.0
    
    def _validate_angle(self, angle: Optional[float]) -> bool:
        """验证角度值是否有效
        
        Args:
            angle: 待验证的角度值
        
        Returns:
            True 如果角度有效，False 否则
        """
        if angle is None:
            return False
        if not np.isfinite(angle):
            return False
        # 角度应该在合理范围内（允许负值，会在归一化时处理）
        return abs(angle) < 1e6
    
    def _compute_adaptive_alpha(self, abs_diff: float) -> float:
        """根据角度变化幅度计算自适应 alpha
        
        Args:
            abs_diff: 角度变化的绝对值（度）
        
        Returns:
            有效的 alpha 值 [0, 1]
        """
        if abs_diff < self.noise_threshold_deg_:
            # 极小变化，视为噪声，减弱更新
            effective_alpha = self.alpha_ * self.noise_alpha_factor_
        elif abs_diff < self.normal_threshold_deg_:
            # 正常缓慢转向，使用原始 alpha
            effective_alpha = self.alpha_
        else:
            # 大幅转向，放大 alpha，加快跟踪
            effective_alpha = min(1.0, self.alpha_ * self.large_turn_alpha_factor_)
        
        return max(0.0, min(1.0, effective_alpha))
    
    def Update_(self, raw_angle: Optional[float]) -> Optional[float]:
        """更新角度，返回平滑后的角度
        
        Args:
            raw_angle: 原始角度（度），如果为 None 则返回当前有效角度
        
        Returns:
            平滑后的角度（度）或 None（如果从未有有效输入）
        """
        # 如果输入为 None，返回当前有效角度（保持连续性）
        if raw_angle is None:
            return self.angle_ if self.valid_ else None
        
        # 验证输入
        if not self._validate_angle(raw_angle):
            logger.warning(f"AngleSmoother: 无效的角度输入 {raw_angle}，使用当前角度")
            return self.angle_ if self.valid_ else None
        
        # 归一化角度到 [0, 360)
        normalized_angle = self._normalize_angle(raw_angle)
        
        # 首次有效输入，直接使用
        if not self.valid_:
            self.angle_ = normalized_angle
            self.valid_ = True
            return self.angle_
        
        # 计算角度差
        diff = self._compute_angle_diff(normalized_angle, self.angle_)
        
        # 限幅：单帧最大变化角度
        abs_diff = abs(diff)
        if abs_diff > self.max_step_deg_:
            diff = self.max_step_deg_ if diff > 0 else -self.max_step_deg_
            abs_diff = self.max_step_deg_
        
        # 计算自适应 alpha
        effective_alpha = self._compute_adaptive_alpha(abs_diff)
        
        # 更新平滑角度
        self.angle_ = self._normalize_angle(self.angle_ + effective_alpha * diff)
        
        return self.angle_
    
    def Reset_(self) -> None:
        """重置平滑器状态"""
        self.angle_ = None
        self.valid_ = False


class MinimapDetector:
    """
    双模型小地图检测器

    检测流程：
    1. 使用 detect_engine_（检测模型）在全图上检测 self_arrow 和 enemy_flag
    2. 如果找到 self_arrow bbox，裁剪后优先使用 pose_engine_（姿态模型）检测关键点
    3. 通过关键点计算精确的位置和朝向（Head=0, Tail=1）
    4. 如果 pose_engine_ 检测失败，回退到 OpencvArrowDetector 检测中心和角度

    输入：一张完整的小地图图像 (frame_minimap)
    输出：self_pos, self_angle, enemy_flag_pos 以及原始检测结果列表
    """

    def __init__(self):
        config = GetGlobalConfig()
        self.conf_threshold_ = config.model.conf_threshold
        self.iou_threshold_ = config.model.iou_threshold
        self.detect_engine_ = DetectionEngine(config.model.base_path)
        self.pose_engine_ = DetectionEngine(config.model.arrow_path)
        self.arrow_detector_ = OpencvArrowDetector()

        # 角度平滑器参数
        angle_cfg = config.angle_detection
        self.angle_smoother_ = AngleSmoother(
            alpha=angle_cfg.smoothing_alpha,
            max_step_deg=angle_cfg.max_step_deg,
            noise_threshold_deg=angle_cfg.noise_threshold_deg,
            normal_threshold_deg=angle_cfg.normal_threshold_deg,
            noise_alpha_factor=angle_cfg.noise_alpha_factor,
            large_turn_alpha_factor=angle_cfg.large_turn_alpha_factor,
        )

        # 基地位置缓存，初始化为 None，只有成功检测到 enemy_flag 后才设置
        self.base_position: Optional[Tuple[float, float]] = None

        # 缓存类别 ID
        # TODO(@liangyu) hack 直接写死
        self.arrow_class_id_: Optional[int] = 0
        self.flag_class_id_: Optional[int] = 1

    # -----------------------------------------------------------
    def LoadModel(self) -> bool:
        """加载模型并初始化类别名称映射"""
        if not self.detect_engine_.LoadModel():
            logger.error("MinimapDetector: detect_engine 模型加载失败")
            return False

        logger.info("MinimapDetector: 模型加载成功")
        return True


    # -----------------------------------------------------------
    def Detect(self, frame: np.ndarray) -> MinimapDetectionResult:
        """双模型检测流程：
        1. 使用 detect_engine_ 在全图上检测 self_arrow 和 enemy_flag
        2. 如果找到 self_arrow bbox，裁剪后优先使用 pose_engine_ 检测关键点（中心和角度）
        3. 如果 pose_engine_ 检测失败，回退到 OpencvArrowDetector 检测中心和角度
        """
        if frame is None or frame.size == 0:
            logger.error("MinimapDetector: frame 为空或无效")
            return MinimapDetectionResult(None, None, None, [])

        # === 第一步：全局检测 ===
        detect_result = self.detect_engine_.Detect(
            frame,
            confidence_threshold=self.conf_threshold_,
            iou_threshold=self.iou_threshold_,
            max_det=10
        )

        if not detect_result or len(detect_result) == 0:
            logger.warning("MinimapDetector: detect_engine 检测结果为空")
            return MinimapDetectionResult(None, None, self.base_position, [])

        # 解析检测结果（用于返回和可视化）
        raw_detections = self._parse_results(detect_result)

        # 检测并缓存 enemy_flag 位置（只需一次）
        # TODO(@liangyu) 在Stop时是否需要重置？
        if self.base_position is None:
            self.base_position = self._extract_center_from_yolo(detect_result, self.flag_class_id_)
            if self.base_position is not None:
                logger.debug(f"检测到 enemy_flag 位置并缓存: {self.base_position}")
            else:
                logger.warning("MinimapDetector: 未能在检测结果中找到 enemy_flag")

        # 查找 self_arrow 的 bbox（YOLO 返回的第一个结果就是 conf 最高的）
        arrow_bbox = self._extract_bbox_from_yolo(detect_result, self.arrow_class_id_)
        if arrow_bbox is None:
            logger.warning("MinimapDetector: 未检测到 self_arrow")
            return MinimapDetectionResult(None, None, self.base_position, raw_detections)
        
        x1, y1, x2, y2 = arrow_bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # === 第二步：裁剪箭头区域 ===
        # 以中心点为基准，裁剪固定大小的区域
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        crop_size = 24
        half_size = crop_size // 2
        
        h, w = frame.shape[:2]
        crop_x1 = int(max(0, cx - half_size))
        crop_y1 = int(max(0, cy - half_size))
        crop_x2 = int(min(w, cx + half_size))
        crop_y2 = int(min(h, cy + half_size))

        # 裁剪图像
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            logger.warning("MinimapDetector: 裁剪区域为空")
            return MinimapDetectionResult(None, None, self.base_position, raw_detections)

        # === 第二步：优先使用 pose_engine_ 检测关键点 ===
        local_center = None
        raw_angle = None
        
        pose_result = self.pose_engine_.Detect(
            crop,
            confidence_threshold=self.conf_threshold_,
            iou_threshold=self.iou_threshold_,
            max_det=1
        )
        
        pose_data = self._parse_pose_result(pose_result)
        if pose_data is not None:
            local_center, raw_angle = pose_data
            logger.debug("MinimapDetector: 使用 pose_engine_ 检测成功")
        else:
            # === 回退：使用 OpenCV 箭头检测 ===
            logger.debug("MinimapDetector: pose_engine_ 检测失败，回退到 OpencvArrowDetector")
            local_center, raw_angle = self.arrow_detector_.detect(crop)
            
            if local_center is None:
                logger.warning("MinimapDetector: OpencvArrowDetector 未能检测到箭头中心")
            if raw_angle is None:
                logger.warning("MinimapDetector: OpencvArrowDetector 未能检测到角度")

        # 转换为全局坐标
        self_pos = None
        if local_center is not None:
            cx_local, cy_local = local_center
            self_pos = (cx_local + crop_x1, cy_local + crop_y1)
        
        # 应用角度平滑
        self_angle = self.angle_smoother_.Update_(raw_angle)

        result = MinimapDetectionResult(
            self_pos=self_pos,
            self_angle=self_angle,
            enemy_flag_pos=self.base_position,
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
            if boxes is None or boxes.shape[0] == 0:
                continue
            
            # Batch conversion to numpy (CPU) to avoid repeated tensor access
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
            color = (0, 255, 0) if cls_id == self.arrow_class_id_ else (0, 0, 255)
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


    def _extract_center_from_yolo(
        self, 
        yolo_results: List, 
        class_id: int
    ) -> Optional[Tuple[float, float]]:
        """从 YOLO Results 中提取指定类别的第一个检测框的中心点
        
        Args:
            yolo_results: YOLO Results 对象列表
            class_id: 类别 ID
        
        Returns:
            中心点坐标 (x, y) 或 None
        """
        for result in yolo_results:
            boxes = result.boxes
            if boxes is None or boxes.shape[0] == 0:
                continue
            
            # 转换为 numpy
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            
            # 查找第一个匹配的类别（YOLO 已按 conf 排序）
            for i in range(len(xyxy)):
                if int(cls_ids[i]) == class_id:
                    x1, y1, x2, y2 = xyxy[i]
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    return (float(center_x), float(center_y))
        
        return None
    
    def _extract_bbox_from_yolo(
        self, 
        yolo_results: List, 
        class_id: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """从 YOLO Results 中提取指定类别的第一个检测框
        
        Args:
            yolo_results: YOLO Results 对象列表
            class_id: 类别 ID
        
        Returns:
            边界框 (x1, y1, x2, y2) 或 None
        """
        for result in yolo_results:
            boxes = result.boxes
            if boxes is None or boxes.shape[0] == 0:
                continue
            
            # 转换为 numpy
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            
            # 查找第一个匹配的类别（YOLO 已按 conf 排序）
            for i in range(len(xyxy)):
                if int(cls_ids[i]) == class_id:
                    x1, y1, x2, y2 = xyxy[i]
                    return (float(x1), float(y1), float(x2), float(y2))
        
        return None

    def _parse_pose_result(
        self, 
        pose_results: List, 
        min_keypoint_conf: float = 0.5
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """解析 Pose 模型的关键点结果，提取箭头中心和角度
        
        Args:
            pose_results: YOLO pose Results 对象列表
            min_keypoint_conf: 关键点最小置信度阈值
        
        Returns:
            (local_center, raw_angle) 或 None
            - local_center: (x, y) 相对于裁剪图像的坐标
            - raw_angle: 角度（度），范围 [0, 360)
        """
        if not pose_results or len(pose_results) == 0:
            return None
        
        # 取第一个结果
        result = pose_results[0]
        
        # 检查是否有 keypoints
        if not hasattr(result, "keypoints") or result.keypoints is None:
            return None
        
        keypoints = result.keypoints
        if keypoints.shape[0] == 0:
            return None
        
        # keypoints 形状通常是 (N, num_keypoints, 3)
        # 取第一个检测到的目标的关键点
        kpts = keypoints.data[0]  # shape: (num_keypoints, 3)
        
        # 检查关键点数量（至少需要 2 个：Head 和 Tail）
        if kpts.shape[0] < 2:
            logger.warning("MinimapDetector: pose 模型关键点数量不足")
            return None
        
        # 转换为 numpy 数组（如果在 GPU 上）
        if hasattr(kpts, "cpu"):
            kpts = kpts.cpu().numpy()
        else:
            kpts = np.array(kpts)
        
        # 提取 Head (index 0) 和 Tail (index 1)
        # 格式: [x, y, confidence]
        head = kpts[0]
        tail = kpts[1]
        
        # 检查关键点置信度
        head_conf = head[2] if len(head) > 2 else 1.0
        tail_conf = tail[2] if len(tail) > 2 else 1.0
        
        if head_conf < min_keypoint_conf or tail_conf < min_keypoint_conf:
            logger.debug(f"MinimapDetector: 关键点置信度不足 (head={head_conf:.2f}, tail={tail_conf:.2f})")
            return None
        
        # 计算中心点（Head 和 Tail 的中点）
        center_x = (head[0] + tail[0]) / 2.0
        center_y = (head[1] + tail[1]) / 2.0
        local_center = (float(center_x), float(center_y))
        
        # 计算角度：Tail -> Head 的方向
        # 向量从 Tail 指向 Head
        dx = head[0] - tail[0]
        dy = head[1] - tail[1]
        
        # 使用 atan2 计算角度（弧度）
        angle_rad = np.arctan2(dy, dx)
        # 转换为度，并归一化到 [0, 360)
        raw_angle = np.degrees(angle_rad) % 360.0
        
        return (local_center, float(raw_angle))

    # -----------------------------------------------------------
    def Reset(self):
        """重置检测器状态"""
        self.angle_smoother_.Reset_()
        self.base_position = None

def run_benchmark(
    image_path: str,
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
    detector = MinimapDetector()
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
        target_fps=args.fps,
        max_frames=max_frames,
        show_visualization=True,
    )
