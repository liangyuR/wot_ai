from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import numpy as np
from loguru import logger

from src.vision.detection_engine import DetectionEngine
from src.navigation.config.loader import load_config
from src.utils.global_path import GetConfigPath


@dataclass
class ScreenTarget:
    """主视野中的一个目标。

    所有坐标均为屏幕像素坐标（相对于当前 frame）。
    """

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (cx, cy)
    score: float                     # YOLO 置信度
    cls_id: int                      # YOLO 类别 id，比如 0=tank_body, 1=enemy_hpbar ...
    is_enemy: bool                   # 是否为敌方目标（基于血条/红色 UI 判断）
    est_distance: Optional[float] = None  # 粗略估算距离（比如用 bbox 高度反推）


@dataclass
class AimReticle:
    """瞄准指示圈（绿色虚线圆圈）的检测结果。"""

    center: Tuple[int, int]
    radius: int
    score: float


@dataclass
class HpBarDet:
    """敌方血条检测结果（用于预瞄）。"""

    bbox: Tuple[int, int, int, int]
    score: float


class MainViewDetector:
    """使用 YOLO 模型检测主视野中的目标 + 瞄准圈 + 敌方血条。

    职责：
    - 调用 YOLO 模型推理
    - 解析输出为 ScreenTarget 列表（敌方坦克）
    - 解析瞄准指示圈 AimReticle
    - 提供敌方血条 HpBarDet 列表（供预瞄使用）
    - 不负责“选哪个目标”，那是上层 AttackTargetSelector / 预瞄逻辑的工作
    """

    # 约定 hpbar/aim_reticle 模型中的类别 id
    CLS_ENEMY_HPBAR = 0
    CLS_AIM_RETICLE = 1

    def __init__(self, conf_threshold: Optional[float] = None) -> None:
        """初始化检测器。

        Args:
            conf_threshold: 置信度过滤阈值。如果为 None，使用配置文件中的值。
        """

        self.config = load_config(GetConfigPath())

        # 加载模型
        self.tank_detector = DetectionEngine(self.config.model.tank_path, "cuda")
        self.hpbar_detector = DetectionEngine(self.config.model.hpbar_path, "cuda")

        # 预热，减少首帧延迟（这里假定主视野约为 1280x720，可按需调整）
        self.tank_detector.Warmup((1280, 720))
        self.hpbar_detector.Warmup((1280, 720))

        # 阈值配置
        self.conf_threshold = conf_threshold or self.config.model.conf_threshold
        self.iou_threshold = self.config.model.iou_threshold

    # ------------------------------------------------------------------ #
    # 对外主接口
    # ------------------------------------------------------------------ #
    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[List[ScreenTarget], Optional[AimReticle], List[HpBarDet]]:
        """在主视野 frame 中检测所有“可攻击目标”、瞄准圈和敌方血条。

        返回：
            targets:   ScreenTarget 列表（已做血条-车身关联，只保留敌方坦克）
            reticle:   AimReticle 或 None（瞄准圈位置和置信度）
            hp_bars:   敌方血条列表（即使没有车身也会返回，可用于预瞄）
        """
        targets: List[ScreenTarget] = []
        reticle: Optional[AimReticle] = None
        hp_bars: List[HpBarDet] = []

        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("MainViewDetector.detect: frame 为空")
            return targets, reticle, hp_bars

        # 1) 运行两个 YOLO 模型
        tank_dets = self._run_tank_detector(frame_bgr)
        ui_dets = self._run_hpbar_and_reticle_detector(frame_bgr)

        # 2) 敌方血条列表（无论是否能关联到车身，都返回给上层做预瞄）
        hp_bars = self._extract_enemy_hpbars(ui_dets)

        # 3) 做血条 -> 车身关联（只有同时有 tank 和 敌方血条 时才有目标）
        if tank_dets and hp_bars:
            targets = self._associate_hpbar_and_tank(tank_dets, ui_dets)

        # 4) 解析瞄准圈（即使当前没有可攻击目标，也可能有瞄准圈）
        if ui_dets:
            reticle = self._parse_reticle(ui_dets)

        return targets, reticle, hp_bars

    # ------------------------------------------------------------------ #
    # YOLO 推理封装
    # ------------------------------------------------------------------ #
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
                dets.append({
                    "bbox": (x1, y1, x2, y2),
                    "score": score,
                    "cls_id": cls_id,
                })
        return dets

    def _run_hpbar_and_reticle_detector(self, frame_bgr: np.ndarray) -> List[dict]:
        """运行 UI YOLO 检测，返回血条 + 瞄准圈等 UI 的 dict 列表。

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
                dets.append({
                    "bbox": (x1, y1, x2, y2),
                    "score": score,
                    "cls_id": cls_id,
                })
        return dets

    # ------------------------------------------------------------------ #
    # 敌方血条提取（供预瞄使用）
    # ------------------------------------------------------------------ #
    def _extract_enemy_hpbars(self, ui_dets: Sequence[dict]) -> List[HpBarDet]:
        """从 UI 检测结果中提取敌方血条列表。"""
        bars: List[HpBarDet] = []
        for d in ui_dets:
            if d.get("cls_id", -1) != self.CLS_ENEMY_HPBAR:
                continue
            x1, y1, x2, y2 = d["bbox"]
            bars.append(HpBarDet(bbox=(x1, y1, x2, y2), score=float(d["score"])))
        return bars

    # ------------------------------------------------------------------ #
    # 瞄准圈解析
    # ------------------------------------------------------------------ #
    def _parse_reticle(self, ui_dets: List[dict]) -> Optional[AimReticle]:
        """从 UI 检测结果中解析瞄准指示圈。

        策略：
        - 仅保留 cls_id == CLS_AIM_RETICLE 的检测框
        - 取置信度最高的一个
        - 以 bbox 外接矩形估算圆心与半径
        """
        filtered = [d for d in ui_dets if d.get("cls_id", -1) == self.CLS_AIM_RETICLE]
        if not filtered:
            return None

        best = max(filtered, key=lambda d: d["score"])
        x1, y1, x2, y2 = best["bbox"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max(x2 - x1, y2 - y1) // 2

        return AimReticle(center=(cx, cy), radius=radius, score=float(best["score"]))

    # ------------------------------------------------------------------ #
    # 血条 - 车身关联
    # ------------------------------------------------------------------ #
    def _associate_hpbar_and_tank(
        self,
        tanks: Sequence[dict],
        ui_dets: Sequence[dict],
        max_vertical_dist: int = 200,
        max_center_dist: int = 150,
    ) -> List[ScreenTarget]:
        """简单的血条 -> 车身匹配：

        - 仅使用 cls_id == CLS_ENEMY_HPBAR 的 UI 检测框
        - 对每个血条，取其中心点 (hx, hy)
        - 向下找最近的 tank bbox
        - 垂直方向距离 & 中心距离在阈值内则认为关联成功
        """
        targets: List[ScreenTarget] = []

        # 只取敌方血条
        filtered_hpbars = [hp for hp in ui_dets if hp.get("cls_id", -1) == self.CLS_ENEMY_HPBAR]

        for hp in filtered_hpbars:
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

            score = min(float(best_tank["score"]), float(hp["score"]))
            cls_id = int(best_tank["cls_id"])  # 这里认为车身类别才是主类别

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

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    @staticmethod
    def _estimate_distance_from_bbox(bbox: Tuple[int, int, int, int]) -> float:
        """非物理精确，只是给攻击模块一个“近/远”的排序指标。

        思路：bbox 高度越大，距离越近。
        返回一个“伪距离”：1 / height，上层根据这个排序即可。
        """
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)
        return 1.0 / float(h)


# ---------------- Demo: 批量图片可视化 ----------------


def _draw_targets_on_image(img: np.ndarray, targets: List[ScreenTarget]) -> np.ndarray:
    """在图像上绘制检测到的目标 bbox + 中心点 + 简单标签。"""
    if img is None:
        return img

    import cv2

    vis = img.copy()
    for t in targets:
        x1, y1, x2, y2 = t.bbox
        cx, cy = t.center

        # 车身 bbox（绿色）
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


def _draw_reticle_on_image(img: np.ndarray, reticle: Optional[AimReticle]) -> np.ndarray:
    """在图像上绘制瞄准圈。"""
    if img is None or reticle is None:
        return img

    import cv2

    vis = img.copy()
    cx, cy = reticle.center
    r = reticle.radius

    cv2.circle(vis, (cx, cy), r, (0, 255, 255), 2)
    cv2.putText(
        vis,
        f"reticle {reticle.score:.2f}",
        (cx + 5, cy + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def _draw_hpbars_on_image(img: np.ndarray, hp_bars: List[HpBarDet]) -> np.ndarray:
    """在图像上绘制敌方血条 bbox（用于调试预瞄）。"""
    if img is None or not hp_bars:
        return img

    import cv2

    vis = img.copy()
    for hp in hp_bars:
        x1, y1, x2, y2 = hp.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            vis,
            f"hp {hp.score:.2f}",
            (x1, max(0, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


if __name__ == "__main__":
    import cv2
    import glob
    import os
    import argparse

    parser = argparse.ArgumentParser(description="MainViewDetector demo")
    parser.add_argument("--img_dir", type=str, default="src/attack/vision/test_images", help="图片目录")
    args = parser.parse_args()

    detector = MainViewDetector()

    img_dir = args.img_dir
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

        targets, reticle, hp_bars = detector.detect(img)

        # 打印检测到的目标信息
        if not targets:
            logger.info("未检测到敌方坦克目标")
        else:
            for t in targets:
                logger.info(
                    "target bbox=%s, center=%s, score=%.3f, est_dist=%s",
                    t.bbox,
                    t.center,
                    t.score,
                    t.est_distance,
                )

        if reticle is None:
            logger.info("未检测到瞄准圈")
        else:
            logger.info(
                "reticle center=%s, radius=%d, score=%.3f",
                reticle.center,
                reticle.radius,
                reticle.score,
            )

        if not hp_bars:
            logger.info("未检测到敌方血条")
        else:
            logger.info("检测到 %d 个敌方血条", len(hp_bars))

        vis = _draw_targets_on_image(img, targets)
        vis = _draw_reticle_on_image(vis, reticle)
        vis = _draw_hpbars_on_image(vis, hp_bars)

        win_name = os.path.basename(img_path)
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(0)
        cv2.destroyWindow(win_name)

        # 按 q 或 ESC 退出
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
