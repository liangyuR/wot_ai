from pathlib import Path
from typing import Optional, Tuple, Union

from loguru import logger
import numpy as np
import time

class TemplateMatcher:
    """
    封装模板匹配与点击功能
    """
    def __init__(self):
        from src.utils.template_repository import TemplateRepository
        self._template_repository = TemplateRepository()

    def click_template(
        self,
        template: Union[str, Path],
        confidence: float = 0.85,
        region: Optional[Tuple[int, int, int, int]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        在屏幕上查找模板，并点击其中心点。
        """
        import pyautogui
        center = self.match_template(template, confidence=confidence, region=region, frame=frame)
        if center is not None:
            pyautogui.click(center[0], center[1])
            time.sleep(0.5)
            pyautogui.click(center[0], center[1])
            return center
        return None

    def match_template(
        self,
        template: Union[str, Path],
        confidence: float = 0.75,
        region: Optional[Tuple[int, int, int, int]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        在屏幕上查找模板，返回匹配位置的中心点坐标。

        Args:
            template: 模板名称、相对路径或绝对路径
            confidence: 匹配置信度（0.0-1.0），默认 0.85
            region: 搜索区域 (left, top, width, height)，None 表示全屏
        """
        tpl_path = Path(template)
        template_path: Optional[Path] = None

        if tpl_path.is_absolute():
            if tpl_path.exists():
                template_path = tpl_path
            else:
                logger.error(f"模板不存在: {tpl_path}")
                return None
        else:
            self._template_repository.ensure_loaded()
            template_path = self._template_repository.get_path(template)
            if template_path is None:
                fallback = self._template_repository.find_on_disk(template)
                if fallback:
                    self._template_repository.register_path(fallback)
                    template_path = fallback

        if template_path is None:
            logger.info("模板未找到: %s", template)
            return None

        if frame is None:
            from src.utils.screen_action import ScreenAction
            frame = ScreenAction.screenshot(region=region)

        if frame is None:
            logger.error(f"截图失败: {region}")
            return None

        import cv2
        # Load template image as grayscale
        tpl_img = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if tpl_img is None:
            logger.error(f"模板图片加载失败: {template_path}")
            return None

        # Convert input frame to grayscale if needed
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            frame_gray = frame

        if tpl_img.ndim == 3 and tpl_img.shape[2] == 3:
            tpl_gray = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY)
        elif tpl_img.ndim == 3 and tpl_img.shape[2] == 4:
            tpl_gray = cv2.cvtColor(tpl_img, cv2.COLOR_BGRA2GRAY)
        else:
            tpl_gray = tpl_img

        # If region is specified, crop the frame accordingly
        search_img = frame_gray
        top_left_offset = (0, 0)
        if region is not None:
            x, y, w, h = region
            search_img = frame_gray[y:y+h, x:x+w]
            top_left_offset = (x, y)

        # Perform template matching
        result = cv2.matchTemplate(search_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        logger.info(f"模板匹配最大置信度：{max_val:.3f}，期望：{confidence}")
        if max_val >= confidence:
            tpl_h, tpl_w = tpl_gray.shape[:2]
            center_x = max_loc[0] + tpl_w // 2 + top_left_offset[0]
            center_y = max_loc[1] + tpl_h // 2 + top_left_offset[1]
            center = (center_x, center_y)
            logger.info(f"找到模板 {template_path} 在位置: {center}")
            return center
        logger.info(f"未在画面中找到模板: {template_path}")
        return None

# 保持与模块初始用法兼容的函数
_template_matcher_instance = TemplateMatcher()
click_template = _template_matcher_instance.click_template
match_template = _template_matcher_instance.match_template


if __name__ == "__main__":
    import cv2
    from src.utils.screen_action import ScreenAction
    frame = ScreenAction.screenshot(region=None)
    template_matcher = TemplateMatcher()
    center = template_matcher.click_template("C:/Users/11601/Desktop/Snipaste_2025-11-20_11-00-11.png", confidence=0.85, region=None, frame=frame)
    if center is not None:
        cv2.circle(frame, center, radius=6, color=(0, 0, 255), thickness=-1)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()