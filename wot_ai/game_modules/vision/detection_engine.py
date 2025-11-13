#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全新检测引擎：封装 YOLO 检测逻辑

设计目标：
- 仅依赖显式传入的 model_path，不再处理 base_dir
- 支持 GPU / CPU，自适应 half(FP16)
- 提供显式 LoadModel() 与可选 Warmup() 接口
- Detect 保持向后兼容：返回 List[Dict]
- 额外提供 GetBestTarget 等实用方法
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from pathlib import Path

import numpy as np
from loguru import logger
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - 运行环境可能没有显式 torch
    torch = None  # type: ignore


@dataclass
class Detection:
    """结构化检测结果（内部使用，可选对外暴露）"""
    cls: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class DetectionEngine:
    """YOLO 检测引擎"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_half: bool = True,
        log_detections: bool = False,
    ) -> None:
        """初始化检测引擎

        Args:
            model_path: YOLO 模型权重路径（相对或绝对）
            device: "auto" | "cpu" | "cuda" | "cuda:0" ...
            use_half: 在支持 GPU 时尝试使用 FP16
            log_detections: 是否逐条打印检测结果（调试用）
        """
        if not model_path:
            raise ValueError("model_path 不能为空")

        self.model_path_ = str(model_path)
        self.device_ = device
        self.use_half_ = use_half
        self.log_detections_ = log_detections

        self.model_: Optional[YOLO] = None
        self.actual_device_: Optional[str] = None
        self.warmed_up_: bool = False

    # ------------------------------------------------------------------
    # 模型加载 & 预热
    # ------------------------------------------------------------------
    def LoadModel(self) -> bool:
        """加载 YOLO 模型

        Returns:
            是否加载成功
        """
        if self.model_ is not None:
            return True

        try:
            model_path_obj = Path(self.model_path_)
            if not model_path_obj.exists():
                logger.error(f"模型文件不存在: {model_path_obj}")
                return False

            logger.info(f"加载 YOLO 模型: {model_path_obj.resolve()}" )
            model = YOLO(str(model_path_obj))

            # 设备选择
            device = self._select_device()
            if device is not None:
                model.to(device)
                self.actual_device_ = str(device)
            else:
                self.actual_device_ = "cpu"

            # FP16
            if self.use_half_ and self._can_use_half():
                try:
                    if hasattr(model, "model") and hasattr(model.model, "half"):
                        model.model.half()
                        logger.info("YOLO 模型已切换为 FP16")
                    else:
                        logger.warning("当前 YOLO 版本不支持 half()，已跳过 FP16 设置")
                except Exception as e:  # pragma: no cover - 环境相关
                    logger.warning(f"设置 FP16 失败，使用 FP32 继续运行: {e}")

            self.model_ = model
            logger.info(f"模型加载成功，device={self.actual_device_}")
            return True
        except Exception as e:  # pragma: no cover - 防御性
            logger.error(f"加载模型失败: {e}")
            self.model_ = None
            return False

    def Warmup(self, img_size: Tuple[int, int] = (640, 640)) -> None:
        """模型预热：跑一帧空白图，避免首帧延迟

        Args:
            img_size: (宽, 高)
        """
        if not self.LoadModel():
            return

        if self.warmed_up_:
            return

        w, h = img_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        try:
            _ = self.model_(dummy, conf=0.01, verbose=False)
            self.warmed_up_ = True
            logger.info(f"模型预热完成，img_size={img_size}")
        except Exception as e:  # pragma: no cover - 防御性
            logger.warning(f"模型预热失败，但不影响后续正常检测: {e}")

    # ------------------------------------------------------------------
    # 核心检测接口
    # ------------------------------------------------------------------
    def Detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict]:
        """检测帧中的目标（向后兼容版）

        Args:
            frame: BGR 格式图像 (H, W, 3)
            confidence_threshold: 置信度阈值
            iou_threshold: NMS 的 IoU 阈值

        Returns:
            检测结果列表，每个元素为 dict：
            - class: 类别 ID
            - confidence: 置信度
            - bbox: [x1, y1, x2, y2]
        """
        detections_struct = self.DetectStructured(
            frame,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

        # 转成旧格式 dict
        results: List[Dict] = []
        for d in detections_struct:
            results.append(
                {
                    "class": int(d.cls),
                    "confidence": float(d.confidence),
                    "bbox": [float(v) for v in d.bbox],
                }
            )
        return results

    def DetectStructured(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """检测帧中的目标（结构化返回）

        Args:
            frame: BGR 格式图像 (H, W, 3)
            confidence_threshold: 置信度阈值
            iou_threshold: NMS 的 IoU 阈值

        Returns:
            List[Detection]
        """
        if frame is None or frame.size == 0:
            logger.error("DetectStructured: 输入帧为空或无效")
            return []

        if not self.LoadModel():
            return []

        try:
            results = self.model_(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False,
            )

            detections: List[Detection] = []

            # YOLO 对单张图像通常返回一个 result
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    det = Detection(
                        cls=cls_id,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                    detections.append(det)

                    if self.log_detections_:
                        name = None
                        try:
                            if hasattr(self.model_, "names"):
                                names = self.model_.names
                                if isinstance(names, dict):
                                    name = names.get(cls_id)
                        except Exception:  # pragma: no cover
                            name = None

                        if name is not None:
                            logger.debug(
                                f"det cls={cls_id}({name}), conf={conf:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
                            )
                        else:
                            logger.debug(
                                f"det cls={cls_id}, conf={conf:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
                            )

            return detections
        except Exception as e:  # pragma: no cover - 防御性
            logger.error(f"检测失败: {e}")
            return []

    # ------------------------------------------------------------------
    # 目标选择工具方法
    # ------------------------------------------------------------------
    def GetBestTarget(
        self,
        detections: List[Dict] | List[Detection],
        target_class: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        """从检测结果中选择最佳目标（返回 bbox 中心点像素坐标）

        Args:
            detections: Detect / DetectStructured 的返回结果
            target_class: 目标类别 ID（若为 None，则从所有检测中选置信度最高者）

        Returns:
            (center_x, center_y) 或 None
        """
        if not detections:
            return None

        # 统一转成 Detection 列表
        if isinstance(detections[0], Detection):
            det_list: List[Detection] = detections  # type: ignore[assignment]
        else:
            det_list = [
                Detection(
                    cls=int(d["class"]),
                    confidence=float(d["confidence"]),
                    bbox=tuple(d["bbox"])  # type: ignore[arg-type]
                )
                for d in detections  # type: ignore[assignment]
            ]

        # 如果指定了类别，先过滤
        if target_class is not None:
            filtered = [d for d in det_list if d.cls == target_class]
            if filtered:
                det_list = filtered

        if not det_list:
            return None

        best = max(det_list, key=lambda d: d.confidence)
        cx, cy = best.center
        return int(cx), int(cy)

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    def _select_device(self) -> Optional[str]:
        """根据配置和环境选择设备字符串"""
        if self.device_ != "auto":
            return self.device_

        # auto 模式
        if torch is None:
            return "cpu"

        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # pragma: no cover
            pass

        return "cpu"

    def _can_use_half(self) -> bool:
        """当前环境是否适合使用 FP16"""
        if not self.use_half_:
            return False

        if torch is None:
            return False

        try:
            return torch.cuda.is_available()
        except Exception:  # pragma: no cover
            return False
