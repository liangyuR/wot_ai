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

    @staticmethod
    def screenshot_with_key_hold(key, hold_duration: float = 4.0, warmup: float = 2.0) -> Optional[np.ndarray]:
        """
        按住指定按键后截图

        Args:
            key: pynput.keyboard.Key 或字符
            hold_duration: 按键保持时间（秒）
            warmup: 按键后等待界面稳定时间（秒）

        Returns:
            BGR格式的numpy数组截图，如果失败则返回None
        """
        from pynput import keyboard
        import time

        controller = keyboard.Controller()
        try:
            controller.press(key)
            time.sleep(warmup)
            remain = max(0.0, hold_duration - warmup)
            time.sleep(remain)
            frame = ScreenAction.screenshot()
            controller.release(key)
            return frame
        except Exception as exc:
            logger.error(f"按住 {key} 截图失败: {exc}")
            return None
        finally:
            try:
                controller.release(key)
            except Exception as rel_exc:
                logger.warning(f"释放按键失败: {key}，错误: {rel_exc}")

if __name__ == "__main__":
    import cv2
    frame = ScreenAction.screenshot_with_key_hold('b')
    cv2.imshow("Screenshot", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()