#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NavigationSession: container for per-match state.

This dataclass holds all state that should be reset between game matches,
while allowing long-lived resources (models, UI) to persist.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NavigationSession:
    """Per-match state container.
    
    All fields here are reset when a match ends, while the NavigationRuntime
    keeps models and UI alive across matches.
    """
    
    # Path planning state
    current_path_grid_: List[Tuple[int, int]] = field(default_factory=list)
    current_path_world_: List[Tuple[float, float]] = field(default_factory=list)
    current_target_idx_: int = 0
    last_published_idx_: int = -1
    
    # Map info
    map_name_: Optional[str] = None
    minimap_region_: Optional[Dict] = None
    
    # Scale factors (grid -> minimap)
    scale_x_: float = 1.0
    scale_y_: float = 1.0
    
    def reset(self) -> None:
        """Reset all per-match state (called at match end)."""
        self.current_path_grid_.clear()
        self.current_path_world_.clear()
        self.current_target_idx_ = 0
        self.last_published_idx_ = -1
    
    def resetFull(self) -> None:
        """Reset everything including map info (called before new match)."""
        self.reset()
        self.map_name_ = None
        self.minimap_region_ = None
        self.scale_x_ = 1.0
        self.scale_y_ = 1.0
    
    def hasValidRegion(self) -> bool:
        """Check if minimap region is configured."""
        return self.minimap_region_ is not None
    
    def getMinimapSize(self) -> Tuple[int, int]:
        """Get minimap size (width, height)."""
        if self.minimap_region_ is None:
            return (0, 0)
        return (self.minimap_region_["width"], self.minimap_region_["height"])
    
    def getMinimapOffset(self) -> Tuple[int, int]:
        """Get minimap offset (x, y)."""
        if self.minimap_region_ is None:
            return (0, 0)
        return (self.minimap_region_["x"], self.minimap_region_["y"])
