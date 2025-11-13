import os
import io
import json
import base64
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from ultralytics import YOLO
import yaml
from loguru import logger

from label_studio_ml.model import LabelStudioMLBase


class YOLO11SegModel(LabelStudioMLBase):
    """
    YOLO11m-seg + Label Studio ML Backend

    功能：
    - __init__: 加载模型 & 基本配置
    - predict: 对任务图片做 seg 推理，返回 bitmap mask（base64 PNG）
    - fit: 预留增量训练入口，目前只做数据收集/打印日志
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open("config.yaml", "r", encoding="utf-8") as f:
            ym_config = yaml.safe_load(f)

        # --- 模型与推理配置 ---
        self.model_path = ym_config["model_path"]
        self.score_threshold = ym_config["score_threshold"]

        # --- Label Studio 配置 ---
        # 这些需要和你的 label config 一致：
        # <Image name="image" ...>
        # <BrushLabels name="segmentation" toName="image">...
        self.from_name = ym_config["from_name"]
        self.to_name = ym_config["to_name"]

        # LS_LABELS 可以是 ["Obstacle", "Drivable", "EnemyTank", ...]
        self.labels = ym_config["labels"]

        # 是否在日志中打印 debug 信息
        self.debug = ym_config["debug"]

        # 加载 YOLO 模型
        self.model = YOLO(self.model_path)

        # debug log: 配置信息和模型加载
        logger.info(f"[YOLO11SegModel] config: {json.dumps(ym_config, ensure_ascii=False)}")
        logger.info(f"[YOLO11SegModel] YOLO model loaded from: {self.model_path}")

        if self.debug:
            print(f"[YOLO11SegModel] Loaded model from {self.model_path}")
            print(f"[YOLO11SegModel] score_threshold={self.score_threshold}")
            print(f"[YOLO11SegModel] labels={self.labels}")
            logger.info(f"[YOLO11SegModel] Debug mode enabled: {self.debug}")

    # ---------------------------------------------------------------------
    # 工具函数：mask -> base64 PNG
    # ---------------------------------------------------------------------
    @staticmethod
    def mask_to_base64_png(mask: np.ndarray) -> str:
        """
        将二值 mask(HxW, 0/1) 转成 base64 编码的 PNG（单通道）
        """
        # 保证是 0/255 的 uint8 图像
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        mask_img = (mask * 255).astype(np.uint8)

        im = Image.fromarray(mask_img, mode="L")
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        b64_str = base64.b64encode(png_bytes).decode("utf-8")
        return b64_str

    # ---------------------------------------------------------------------
    # predict: 供 Label Studio 做预标注
    # ---------------------------------------------------------------------
    def predict(self, tasks: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        tasks: List[task_dict]
        return: List[prediction_dict]
        """
        predictions: List[Dict[str, Any]] = []

        for task in tasks:
            # 1. 拿到本地图片路径（LabelStudioMLBase 提供工具函数）
            image_url = task["data"].get(self.to_name)
            image_path = self.get_local_path(image_url)

            if self.debug:
                print(f"[predict] task id={task.get('id')}, image={image_path}")

            # 2. YOLO 推理
            results = self.model(image_path, verbose=False)
            res = results[0]

            h, w = res.orig_shape  # 原图尺寸
            boxes = res.boxes
            masks = res.masks  # ultralytics YOLO seg：n_masks x H x W

            task_results: List[Dict[str, Any]] = []

            if boxes is None or masks is None:
                # 没检测到，返回空 result
                predictions.append({"result": task_results})
                continue

            num_instances = len(boxes)
            if self.debug:
                print(f"[predict] num_instances={num_instances}")

            for i in range(num_instances):
                box = boxes[i]
                score = float(box.conf.item())
                if score < self.score_threshold:
                    continue

                cls_id = int(box.cls.item())
                # label name：如果你传了 LS_LABELS 就用名字，否则就用 class id 字符串
                if 0 <= cls_id < len(self.labels):
                    label_name = self.labels[cls_id]
                else:
                    label_name = str(cls_id)

                mask_tensor = masks.data[i]  # shape: H x W
                mask_np = mask_tensor.cpu().numpy().astype(np.float32)

                # 3. mask -> base64 PNG
                mask_png_b64 = self.mask_to_base64_png(mask_np)

                # 4. 组装 Label Studio bitmap mask 结果
                # type 使用 "brushlabels"，value 里用 format=png + image=base64
                result_item = {
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "brushlabels",
                    "value": {
                        "format": "png",
                        "image": mask_png_b64,
                        "brushlabels": [label_name],
                        # 可选：有的前端配置会用到宽高
                        "height": h,
                        "width": w,
                    },
                    "score": score,
                }

                task_results.append(result_item)

            predictions.append({"result": task_results})

        return predictions

    # ---------------------------------------------------------------------
    # fit: 预留增量训练接口（这里先实现为“收集数据 + 打印日志”）
    # ---------------------------------------------------------------------
    def fit(
        self,
        tasks: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        用于自动增量训练的入口。

        当前实现：
        - 遍历 tasks + annotations，收集一些统计信息
        - 预留你之后接 YOLO 训练脚本的位置
        - 返回一个简单的 summary，告诉 Label Studio “训练已完成”
        """

        # 这里仅做最小实现，不真正启动训练
        num_samples = len(tasks)
        num_annotations = len(annotations)

        if self.debug:
            print("[fit] called")
            print(f"[fit] num_samples={num_samples}, num_annotations={num_annotations}")

        # 示例：你可以在这里把标注保存成 YOLO 的训练数据格式
        # dataset_root = os.getenv("TRAIN_DATASET_DIR", "./ls_yolo_seg_dataset")
        # os.makedirs(dataset_root, exist_ok=True)
        # 在这里遍历 tasks + annotations，把 bitmap / polygon 转换成 YOLO seg 标签文件
        # 然后调用：
        #   train_model = YOLO(self.model_path)
        #   train_model.train(data="your_dataset.yaml", epochs=..., imgsz=..., ...)
        #
        # 这一部分我先留空，让你根据自己项目结构（wot_ai 中已有的训练脚本）接进去。

        # fit 必须返回一个 dict，Label Studio 用它来展示训练状态
        return {
            "status": "ok",
            "detail": "fit() stub called, no actual training implemented yet.",
            "num_samples": num_samples,
            "num_annotations": num_annotations,
        }
