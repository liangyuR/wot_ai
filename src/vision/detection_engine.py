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

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

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
        if self.model_ is not None:
            return True
    
        try:
            model_path_obj = Path(self.model_path_)
            if not model_path_obj.exists():
                logger.error(f"模型文件不存在: {model_path_obj}")
                return False
    
            logger.info(f"加载 YOLO 模型: {model_path_obj.resolve()}")
            model = YOLO(str(model_path_obj))
    
            # ===== 设备选择：优先用 cuda:0 =====
            if self.device_ == "auto":
                if torch.cuda.is_available():
                    device = "cuda:0"
                else:
                    device = "cpu"
            else:
                device = self.device_
    
            logger.info(f"DetectionEngine 期望使用 device={device}")
            model.to(device)
            self.actual_device_ = str(device)
    
            # ===== 开 half() =====
            self._use_fp16_runtime_ = False
            if self._can_use_half():
                try:
                    if hasattr(model, "model") and hasattr(model.model, "half"):
                        model.model.half()
                        self._use_fp16_runtime_ = True
                        logger.info("YOLO 模型已切换为 FP16 (half precision)")
                except Exception as e:
                    logger.warning(f"设置 FP16 失败，退回 FP32: {e}")
    
            self.model_ = model
    
            # 打印实际 device
            if hasattr(model, "device"):
                actual_device_str = str(model.device)
            else:
                actual_device_str = self.actual_device_
    
            logger.info(f"模型加载成功，实际 device={actual_device_str}, use_half={self._use_fp16_runtime_}")
            if "cpu" in actual_device_str.lower():
                logger.warning("⚠️ YOLO模型实际运行在CPU上，性能会显著下降")
    
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model_ = None
            return False
    # def LoadModel(self) -> bool:
    #     """加载 YOLO 模型

    #     Returns:
    #         是否加载成功
    #     """
    #     if self.model_ is not None:
    #         return True

    #     try:
    #         model_path_obj = Path(self.model_path_)
    #         if not model_path_obj.exists():
    #             logger.error(f"模型文件不存在: {model_path_obj}")
    #             return False

    #         logger.info(f"加载 YOLO 模型: {model_path_obj.resolve()}" )
    #         model = YOLO(str(model_path_obj))

    #         # 设备选择
    #         device = self._select_device()
    #         if device is not None:
    #             model.to(device)
    #             self.actual_device_ = str(device)
    #         else:
    #             self.actual_device_ = "cpu"

    #         # FP16
    #         if self.use_half_ and self._can_use_half():
    #             try:
    #                 if hasattr(model, "model") and hasattr(model.model, "half"):
    #                     # model.model.half()
    #                     logger.info("YOLO 模型已切换为 FP16")
    #                 else:
    #                     logger.warning("当前 YOLO 版本不支持 half()，已跳过 FP16 设置")
    #             except Exception as e:  # pragma: no cover - 环境相关
    #                 logger.warning(f"设置 FP16 失败，使用 FP32 继续运行: {e}")

    #         self.model_ = model
    #         # 检查实际设备并记录
    #         if hasattr(model, 'device'):
    #             actual_device_str = str(model.device) if hasattr(model.device, '__str__') else str(model.device)
    #         else:
    #             actual_device_str = self.actual_device_
            
    #         logger.info(f"模型加载成功，device={actual_device_str}")
    #         if "cpu" in actual_device_str.lower():
    #             logger.warning("⚠️ YOLO模型运行在CPU上，性能会显著下降（预期80-200ms vs GPU的3-6ms）")
    #         return True
    #     except Exception as e:  # pragma: no cover - 防御性
    #         logger.error(f"加载模型失败: {e}")
    #         self.model_ = None
    #         return False

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
        iou_threshold: float = 0.25,
        max_det: int = 100,  # 添加此参数，限制最大检测数量
    ) -> List[Dict]:
        """检测帧中的目标（向后兼容版）

        Args:
            frame: BGR 格式图像 (H, W, 3)
            confidence_threshold: 置信度阈值
            iou_threshold: NMS 的 IoU 阈值
            max_det: 最大检测数量，限制NMS处理时间

        Returns:
            检测结果列表，每个元素为 dict：
            - class: 类别 ID
            - confidence: 置信度
            - bbox: [x1, y1, x2, y2]
        """
        if frame is None or frame.size == 0:
            logger.error("输入帧为空或无效")
            return []

        if not self.LoadModel():
            logger.error("模型加载失败")
            return []

        try:
            return self.model_(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_det,  # 添加此参数
                verbose=False,
            )
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []

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
