from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

GridCoord = Tuple[int, int]

@dataclass
class MapModel:
    grid: np.ndarray                 # 0/1 栅格：1=障碍
    cost_map: Optional[np.ndarray]   # 代价图（np.inf = 障碍）
    inflated_obstacle: Optional[np.ndarray]  # 0/1 膨胀障碍
    size: Tuple[int, int]            # (grid_w, grid_h)

@dataclass
class PlanRequest:
    start: GridCoord
    goal: GridCoord

@dataclass
class PlanResult:
    ok: bool
    path: List[GridCoord]
    reason: str = ""
