from loguru import logger
from typing import List, Sequence, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from src.vision.detection_engine import DetectionEngine
from src.navigation.config.loader import load_config
from src.utils.global_path import GetConfigPath


@dataclass
class ScreenTarget:
    """主视野中的一个攻击目标（敌方坦克）。

    所有坐标均为屏幕像素坐标（相对于当前 frame）。
    """

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (cx, cy)
    score: float  # YOLO 置信度
    cls_id: int  # YOLO 类别 id，比如 0=tank_body, 1=enemy_hpbar ...
    is_enemy: bool  # 是否为敌方（基于血条/红色UI判断）
    # 可选的一些辅助信息，后续可以慢慢加：
    est_distance: Optional[float] = None  # 粗略估算距离（比如用 bbox 高度反推）


class MainViewDetector:
    """使用 YOLO 模型检测主视野中的敌方坦克目标。

    职责：
    - 调用 YOLO 模型推理
    - 解析输出为 ScreenTarget 列表
    - 做基础过滤（置信度阈值、屏幕边缘过滤等）
    - 不负责“选哪个目标”，那是上层 AttackTargetSelector 的活
    """

    def __init__(self, conf_threshold: Optional[float] = None) -> None:
        """初始化检测器。

        Args:
            conf_threshold: 置信度过滤阈值。如果为 None，使用配置文件中的值。
        """

        self.config = load_config(GetConfigPath())

        # 加载模型
        self.tank_detector = DetectionEngine(self.config.model.tank_path, "cuda")
        self.hpbar_detector = DetectionEngine(self.config.model.hpbar_path, "cuda")

        # 预热，减少首帧延迟
        self.tank_detector.Warmup((1280, 720))
        self.hpbar_detector.Warmup((1280, 720))

        # 阈值配置
        self.conf_threshold = conf_threshold or self.config.model.conf_threshold
        self.iou_threshold = self.config.model.iou_threshold

    def detect(self, frame_bgr: np.ndarray) -> List[ScreenTarget]:
        """在主视野 frame 中检测所有“可攻击目标”。

        返回：
            ScreenTarget 列表（已经做了血条-车身关联，只保留被判定为敌人的目标）
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("MainViewDetector.detect: frame 为空")
            return []

        # 1) 运行两个 YOLO 模型
        tank_dets = self._run_tank_detector(frame_bgr)
        hpbar_dets = self._run_hpbar_detector(frame_bgr)

        if not tank_dets or not hpbar_dets:
            return []

        # 2) 做血条 -> 车身关联
        targets = self._associate_hpbar_and_tank(tank_dets, hpbar_dets)

        return targets

    # ---------------- YOLO 推理封装 ----------------

    def _run_tank_detector(self, frame_bgr: np.ndarray) -> List[dict]:
        """运行车身 YOLO 检测，返回 dict 列表。

        每个元素格式：
            { 'bbox': (x1, y1, x2, y2), 'score': float, 'cls_id': int }
        """
        results = self.tank_detector.Detect(
            frame_bgr,
            confidence_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        dets: List[dict] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                score = float(b.conf.item())
                if score < self.conf_threshold:
                    continue
                cls_id = int(b.cls.item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                dets.append({"bbox": (x1, y1, x2, y2), "score": score, "cls_id": cls_id})
        return dets

    def _run_hpbar_detector(self, frame_bgr: np.ndarray) -> List[dict]:
        """运行血条 YOLO 检测，返回 dict 列表。

        每个元素格式：
            { 'bbox': (x1, y1, x2, y2), 'score': float, 'cls_id': int }
        """
        results = self.hpbar_detector.Detect(
            frame_bgr,
            confidence_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        dets: List[dict] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                score = float(b.conf.item())
                if score < self.conf_threshold:
                    continue
                cls_id = int(b.cls.item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                dets.append({"bbox": (x1, y1, x2, y2), "score": score, "cls_id": cls_id})
        return dets

    # ---------------- 关联逻辑 ----------------

    def _associate_hpbar_and_tank(
        self,
        tanks: Sequence[dict],
        hpbars: Sequence[dict],
        max_vertical_dist: int = 200,
        max_center_dist: int = 150,
    ) -> List[ScreenTarget]:
        """简单的血条 -> 车身匹配：

        - 对每个血条，取其中心点 (hx, hy)
        - 向下找最近的 tank bbox
        - 垂直方向距离 & 中心距离在阈值内则认为关联成功
        """
        targets: List[ScreenTarget] = []

        for hp in hpbars:
            hx1, hy1, hx2, hy2 = hp["bbox"]
            hcx = (hx1 + hx2) // 2
            hcy = (hy1 + hy2) // 2

            best_tank = None
            best_dist = 1e9

            for tk in tanks:
                tx1, ty1, tx2, ty2 = tk["bbox"]
                tcx = (tx1 + tx2) // 2
                tcy = (ty1 + ty2) // 2

                # 血条应该大致在车身上方
                if tcy <= hcy:
                    continue

                dy = tcy - hcy
                dx = abs(tcx - hcx)
                center_dist = float((dx * dx + dy * dy) ** 0.5)

                if dy > max_vertical_dist:
                    continue
                if center_dist > max_center_dist:
                    continue

                if center_dist < best_dist:
                    best_dist = center_dist
                    best_tank = tk

            if best_tank is None:
                continue

            # 合并血条 & 车身信息，构造 ScreenTarget
            tx1, ty1, tx2, ty2 = best_tank["bbox"]
            tcx = (tx1 + tx2) // 2
            tcy = (ty1 + ty2) // 2

            score = min(best_tank["score"], hp["score"])
            cls_id = best_tank["cls_id"]  # 这里认为车身类别才是主类别

            # 粗略估算一下“距离”：比如用 bbox 高度的倒数
            est_distance = self._estimate_distance_from_bbox(best_tank["bbox"])

            targets.append(
                ScreenTarget(
                    bbox=(tx1, ty1, tx2, ty2),
                    center=(tcx, tcy),
                    score=score,
                    cls_id=cls_id,
                    is_enemy=True,  # 有红血条就是敌人
                    est_distance=est_distance,
                )
            )

        return targets

    @staticmethod
    def _estimate_distance_from_bbox(bbox: Tuple[int, int, int, int]) -> float:
        """非物理精确，只是给攻击模块一个“近/远”的排序指标。

        思路：bbox 高度越大，距离越近。
        可以返回一个 “伪距离”：1 / height，然后上层根据这个排序。
        """
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)
        # 越大越近，我们就简单返回 1.0 / h
        return 1.0 / float(h)


# ---------------- Demo: 批量图片可视化 ----------------


def _draw_targets_on_image(img: np.ndarray, targets: List[ScreenTarget]) -> np.ndarray:
    """在图像上绘制检测到的目标 bbox + 中心点 + 简单标签。"""
    if img is None:
        return img

    vis = img.copy()
    for t in targets:
        x1, y1, x2, y2 = t.bbox
        cx, cy = t.center

        # 车身 bbox（绿色）
        import cv2

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 中心点（红点）
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

        # 文本：score + 伪距离
        label = f"{t.score:.2f}"
        if t.est_distance is not None:
            label += f" d={t.est_distance:.4f}"

        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


if __name__ == "__main__":
    import cv2
    import glob
    import os

    # 循环读取目录下的图片，依次检测，并将检测结果显示在图片中。
    detector = MainViewDetector()

    # TODO: 按你的项目实际路径修改此目录
    img_dir = "src/attack/vision/test_images"
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]

    img_paths: list[str] = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(img_dir, p)))

    img_paths.sort()

    if not img_paths:
        logger.error(f"未找到测试图片: {img_dir} 下无 png/jpg/jpeg/bmp 文件")

    for img_path in img_paths:
        logger.info(f"处理图片: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"无法读取图片: {img_path}")
            continue

        targets = detector.detect(img)

        # 打印检测到的目标信息
        if not targets:
            logger.info("未检测到目标")
        else:
            for t in targets:
                logger.info(
                    f"target bbox={t.bbox}, center={t.center}, score={t.score:.3f}, est_dist={t.est_distance}"
                )

        vis = _draw_targets_on_image(img, targets)

        win_name = os.path.basename(img_path)
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(0)
        cv2.destroyWindow(win_name)

        # 按 q 或 ESC 退出
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
