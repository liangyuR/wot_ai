import numpy as np
from typing import Optional, Tuple
from loguru import logger

class ScreenAction:
    """
    屏幕操作类：封装截图功能
    """

    @staticmethod
    def screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        截取屏幕截图

        Args:
            region: 可选区域 (left, top, width, height)，None表示全屏

        Returns:
            BGR格式的numpy数组，如果失败则返回None
        """
        import mss
        import cv2
        try:
            with mss.mss() as sct:
                if region is None:
                    # 全屏截图
                    monitor = sct.monitors[1]  # 主显示器
                    screenshot = sct.grab(monitor)
                else:
                    # 区域截图
                    left, top, width, height = region
                    monitor = {
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height
                    }
                    screenshot = sct.grab(monitor)

                # 转换为numpy数组（BGR格式）
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None