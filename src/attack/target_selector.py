from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger

from src.attack.main_view_detector import ScreenTarget


@dataclass
class _TrackedTarget:
    """内部使用：记录当前锁定目标的一些信息."""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    score: float
    est_distance: Optional[float]
    sticky_left: int  # 还剩多少帧的“粘滞期”


class AttackTargetSelector:
    """
    从当前帧的 ScreenTarget 列表中挑选“主攻击目标”。

    规则：
    - 优先屏幕中央附近的目标
    - 优先“看起来更近”的目标（est_distance 越大越近）
    - 优先置信度更高的目标
    - 带有简单的粘滞逻辑，避免目标频繁切换
    """

    def __init__(
        self,
        center_tolerance_px: int = 250,
        sticky_frames: int = 10,
        no_target_forget_frames: int = 3,
    ) -> None:
        """
        Args:
            center_tolerance_px: 离屏幕中心超过这个距离的目标会被惩罚
            sticky_frames: 当锁定一个目标后，至少坚持多少帧不轻易换目标
            no_target_forget_frames: 连续多少帧都没有检测到任何目标就清空锁定
        """
        self.center_tolerance_px = max(1, center_tolerance_px)
        self.sticky_frames = max(0, sticky_frames)
        self.no_target_forget_frames = max(1, no_target_forget_frames)

        self._tracked: Optional[_TrackedTarget] = None
        self._no_target_frames: int = 0

    # ------------------------------------------------------------------ #
    # 对外主接口
    # ------------------------------------------------------------------ #
    def update(
        self,
        targets: List[ScreenTarget],
        screen_center: Tuple[int, int],
    ) -> Optional[ScreenTarget]:
        """
        根据当前检测出的目标和屏幕中心，返回本帧选择的主目标。

        Args:
            targets: 当前帧检测到的所有 ScreenTarget
            screen_center: (cx, cy) 屏幕中心

        Returns:
            选中的 ScreenTarget 或 None（没目标）
        """
        cx, cy = screen_center

        if not targets:
            self._no_target_frames += 1
            if self._no_target_frames >= self.no_target_forget_frames:
                self._tracked = None
            return None

        # 重置“无目标帧计数”
        self._no_target_frames = 0

        # 1) 所有目标计算综合评分
        scored_targets: List[tuple[float, ScreenTarget, float]] = []
        for t in targets:
            tx, ty = t.center
            dx = tx - cx
            dy = ty - cy
            dist_center = (dx * dx + dy * dy) ** 0.5

            score = self._score_target(t, dist_center)

            scored_targets.append((score, t, dist_center))

        # 2) 找出评分最高的候选
        scored_targets.sort(key=lambda x: x[0], reverse=True)
        best_score, best_target, best_dist_center = scored_targets[0]

        # 3) 粘滞逻辑：如果之前有锁定目标，且当前帧还在附近，就尽量不换
        if self._tracked is not None:
            keep = self._should_keep_tracked(
                scored_targets,
                best_score=best_score,
                screen_center=screen_center,
            )
            if keep:
                # 找到与 tracked 最接近的那个 ScreenTarget 当作当前输出
                tracked_target = self._find_closest_to_tracked(targets)
                if tracked_target is not None:
                    self._update_tracked(tracked_target)
                    return tracked_target

        # 4) 否则，锁定新的 best_target
        self._tracked = _TrackedTarget(
            center=best_target.center,
            bbox=best_target.bbox,
            score=best_target.score,
            est_distance=best_target.est_distance,
            sticky_left=self.sticky_frames,
        )
        return best_target

    # ------------------------------------------------------------------ #
    # 内部：评分 & 粘滞判定
    # ------------------------------------------------------------------ #
    def _score_target(self, t: ScreenTarget, dist_center: float) -> float:
        """
        目标评分逻辑：

        - 离屏幕中心越近越好（主导因素）
        - est_distance 越大越好（代表越近）
        - 置信度越高越好
        """
        # 避免除 0
        center_term = 1.0 / (1.0 + dist_center / self.center_tolerance_px)

        # 距离项：est_distance 可能为 None
        if t.est_distance is not None:
            distance_term = t.est_distance  # 你之前定义的是“高度倒数”，越大越近
        else:
            distance_term = 0.0

        conf_term = t.score

        # 简单线性加权，可以以后调：
        # 权重比：中心 0.6，距离 0.25，置信度 0.15
        score = 0.6 * center_term + 0.25 * distance_term + 0.15 * conf_term
        return float(score)

    def _should_keep_tracked(
        self,
        scored_targets: List[tuple[float, ScreenTarget, float]],
        best_score: float,
        screen_center: Tuple[int, int],
    ) -> bool:
        """
        根据当前得分情况 & tracked 状态，判断是否继续保持当前锁定目标。
        """
        if self._tracked is None:
            return False

        # 如果已经没有粘滞帧了，允许自由切换
        if self._tracked.sticky_left <= 0:
            return False

        # 找到一个与 tracked 最接近的目标，看它得分如何
        closest_target = self._find_closest_to_tracked([st for _, st, _ in scored_targets])
        if closest_target is None:
            return False

        # 计算这个“与 tracked 最近”的目标得分
        cx, cy = screen_center
        tx, ty = closest_target.center
        dist_center = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
        tracked_like_score = self._score_target(closest_target, dist_center)

        # 如果这个“tracked 对应目标”的得分和当前 best 差不多，就保持不换
        # 允许 best 比它高一点点（例如 10%）
        if tracked_like_score >= best_score * 0.9:
            self._tracked.sticky_left -= 1
            return True

        return False

    def _find_closest_to_tracked(
        self,
        targets: List[ScreenTarget],
    ) -> Optional[ScreenTarget]:
        """在当前 targets 中找一个与 tracked.center 最近的目标。"""
        if self._tracked is None or not targets:
            return None

        tx, ty = self._tracked.center
        best_t: Optional[ScreenTarget] = None
        best_d = 1e9

        for t in targets:
            cx, cy = t.center
            d = (cx - tx) ** 2 + (cy - ty) ** 2
            if d < best_d:
                best_d = d
                best_t = t

        return best_t

    def _update_tracked(self, t: ScreenTarget) -> None:
        """用当前帧的目标更新 tracked 状态。"""
        self._tracked.center = t.center
        self._tracked.bbox = t.bbox
        self._tracked.score = t.score
        self._tracked.est_distance = t.est_distance
        # 粘滞帧数递减在 _should_keep_tracked 里处理
