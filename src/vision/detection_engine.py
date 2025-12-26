#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全新检测引擎：封装 YOLO 检测逻辑

设计目标：
- 仅依赖显式传入的 model_path，不再处理 base_dir
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

    # CUDA 错误恢复配置
    MAX_CUDA_ERRORS_BEFORE_RESTART = 3   # 连续失败次数后重启程序
    MAX_CUDA_ERRORS_BEFORE_RELOAD = 1    # 连续失败次数后尝试重载模型

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        log_detections: bool = False,
    ) -> None:
        """初始化检测引擎

        Args:
            model_path: YOLO 模型权重路径（相对或绝对）
            device: "auto" | "cpu" | "cuda" | "cuda:0" ...
            log_detections: 是否逐条打印检测结果（调试用）
        """
        if not model_path:
            raise ValueError("model_path 不能为空")

        self.model_path_ = str(model_path)
        self.device_ = device
        self.log_detections_ = log_detections

        self.model_: Optional[YOLO] = None
        self.actual_device_: Optional[str] = None
        self.warmed_up_: bool = False

        # CUDA 错误恢复状态
        self._cuda_error_count: int = 0

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
            self.model_ = model

            # 打印实际 device
            if hasattr(model, "device"):
                actual_device_str = str(model.device)
            else:
                actual_device_str = self.actual_device_

            logger.info(f"模型加载成功，实际 device={actual_device_str}")
            if "cpu" in actual_device_str.lower():
                logger.warning("⚠️ YOLO模型实际运行在CPU上，性能会显著下降")

            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model_ = None
            return False

    # ------------------------------------------------------------------
    # CUDA 错误恢复
    # ------------------------------------------------------------------
    def _isCudaError(self, error: Exception) -> bool:
        """判断是否为 CUDA 相关错误"""
        error_str = str(error).lower()
        cuda_keywords = ["cuda", "gpu", "device", "cudnn", "cublas", "nccl"]
        return any(kw in error_str for kw in cuda_keywords)

    def _handleCudaError(self, error: Exception) -> bool:
        """处理 CUDA 错误，尝试恢复
        
        Returns:
            True 如果应该重试，False 如果应该放弃
        """
        self._cuda_error_count += 1
        logger.warning(f"CUDA 错误 #{self._cuda_error_count}: {error}")

        # 策略1：首次错误，尝试清理 CUDA 缓存并重载模型
        if self._cuda_error_count <= self.MAX_CUDA_ERRORS_BEFORE_RELOAD:
            logger.info("尝试恢复：清理 CUDA 缓存并重载模型...")
            try:
                self._cleanupCuda()
                self.model_ = None  # 强制重载
                if self.LoadModel():
                    logger.info("CUDA 恢复成功，模型已重载")
                    return True
            except Exception as e:
                logger.error(f"CUDA 恢复失败: {e}")

        # 策略2：多次失败后，重启程序
        if self._cuda_error_count >= self.MAX_CUDA_ERRORS_BEFORE_RESTART:
            logger.error(f"连续 {self._cuda_error_count} 次 CUDA 错误，CUDA 无法恢复，重启程序...")
            self._restartProgram()

        return False

    def _cleanupCuda(self, clear_cache: bool = True) -> None:
        """清理 CUDA 资源
        
        Args:
            clear_cache: 是否调用 torch.cuda.empty_cache()，
                         批量清理多个 engine 时可设为 False，最后统一清理
        """
        try:
            if self.model_ is not None:
                # 尝试将模型移到 CPU 再删除
                try:
                    self.model_.to("cpu")
                except Exception:
                    pass
                self.model_ = None

            # 清理 PyTorch CUDA 缓存
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("CUDA cache cleared for this engine")
        except Exception as e:
            logger.debug(f"清理 CUDA 资源时出错: {e}")

    def _restartProgram(self) -> None:
        """重启程序（带 --auto-start 参数）"""
        import sys
        import os

        logger.info("正在重启程序...")

        # 获取当前 Python 解释器和脚本路径
        python = sys.executable
        script = sys.argv[0]
        args = sys.argv[1:]

        # 添加 --auto-start 参数（如果不存在）
        if "--auto-start" not in args:
            args.append("--auto-start")

        # 记录重启信息
        logger.info(f"重启命令: {python} {script} {' '.join(args)}")

        # 使用 os.execv 替换当前进程（不会返回）
        try:
            os.execv(python, [python, script] + args)
        except Exception as e:
            logger.error(f"重启失败: {e}，尝试退出程序...")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 核心检测接口
    # ------------------------------------------------------------------
    def Detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.25,
        max_det: int = 100,
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
            result = self.model_(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=False,
            )
            # 成功，重置错误计数
            self._cuda_error_count = 0
            return result
        except Exception as e:
            if self._isCudaError(e):
                if self._handleCudaError(e):
                    # 恢复成功，重试一次
                    return self.Detect(frame, confidence_threshold, iou_threshold, max_det)
            logger.error(f"检测失败: {e}")
            return []

    def Track(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.25,
        max_det: int = 100,
        persist: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> List:
        """追踪帧中的目标（支持跨帧 ID 持久化）

        Args:
            frame: BGR 格式图像 (H, W, 3)
            confidence_threshold: 置信度阈值
            iou_threshold: NMS 的 IoU 阈值
            max_det: 最大检测数量
            persist: 是否在帧之间保持追踪（设为 True 以维护 track ID）
            tracker: 追踪器配置文件（"bytetrack.yaml" 或 "botsort.yaml"）

        Returns:
            YOLO Results 对象列表，包含追踪 ID
        """
        if frame is None or frame.size == 0:
            logger.error("输入帧为空或无效")
            return []

        if not self.LoadModel():
            logger.error("模型加载失败")
            return []

        try:
            result = self.model_.track(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_det,
                persist=persist,
                tracker=tracker,
                verbose=False,
            )
            # 成功，重置错误计数
            self._cuda_error_count = 0
            return result
        except Exception as e:
            if self._isCudaError(e):
                if self._handleCudaError(e):
                    # 恢复成功，重试一次
                    return self.Track(frame, confidence_threshold, iou_threshold, max_det, persist, tracker)
            logger.error(f"追踪失败: {e}")
            return []

    def ResetTracker(self) -> None:
        """重置追踪器状态（切换场景或重新开始追踪时调用）"""
        if self.model_ is not None:
            try:
                self.model_.predictor = None
            except Exception as e:
                logger.debug(f"重置追踪器: {e}")

    def ResetCudaState(self, clear_cache: bool = False) -> None:
        """手动重置 CUDA 状态（可在战斗结束后调用，清理显存）
        
        Args:
            clear_cache: 是否调用 torch.cuda.empty_cache()，
                         批量清理多个 engine 时可设为 False，最后统一清理
        """
        self._cuda_error_count = 0
        self._cleanupCuda(clear_cache=clear_cache)