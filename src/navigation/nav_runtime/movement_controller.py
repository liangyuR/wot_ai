#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional
from loguru import logger


class MovementController:
    """
    MovementController：高层移动控制器

    对 NavigationExecutor 进行封装，提供更清晰的 API：

        - goto(target_pos, current_pos, heading)
        - stop()
        - maintain_forward()

    目的：
    - 隔离 WASD 逻辑
    - 隔离旋转/前进的细节
    - 控制线程只写“逻辑”，不管低层按键
    """

    def __init__(self, nav_executor):
        self.nav = nav_executor

    def goto(
        self,
        target_pos: Tuple[float, float],
        current_pos: Tuple[float, float],
        heading: float
    ) -> None:
        """
        朝向目标点并保持前进
        """
        try:
            self.nav.RotateToward(target_pos, current_pos, heading)
            self.nav.EnsureMovingForward()
        except Exception as e:
            logger.error(f"MovementController.goto 失败: {e}")

    def stop(self) -> None:
        """完全停止移动"""
        try:
            self.nav.StopMoving()
            self.nav.Stop()
        except Exception as e:
            logger.error(f"MovementController.stop 失败: {e}")

    def maintain_forward(self) -> None:
        """保持前进（若没有前进则启动）"""
        try:
            self.nav.EnsureMovingForward()
        except Exception as e:
            logger.error(f"MovementController.maintain_forward 失败: {e}")
