"""
Windows 窗口捕获工具
通过进程名称或窗口标题自动定位并捕获游戏窗口
"""

import numpy as np
from typing import Optional, Tuple, List
from loguru import logger

try:
    import win32gui
    import win32ui
    import win32con
    import win32process
    from PIL import Image
    WIN32_AVAILABLE = True
except ImportError:
    logger.warning("pywin32 not available, window capture will not work")
    WIN32_AVAILABLE = False


class WindowCapture:
    """
    Windows 窗口捕获类
    
    支持通过进程名称或窗口标题自动定位目标窗口
    """
    
    def __init__(
        self,
        process_name: Optional[str] = None,
        window_title: Optional[str] = None,
        partial_match: bool = True
    ):
        """
        初始化窗口捕获器
        
        Args:
            process_name: 进程名称（如 "WorldOfTanks.exe"）
            window_title: 窗口标题（如 "World of Tanks"）
            partial_match: 是否使用部分匹配（推荐）
        """
        if not WIN32_AVAILABLE:
            raise ImportError("pywin32 is required for window capture. Install: pip install pywin32")
        
        self.process_name_ = process_name
        self.window_title_ = window_title
        self.partial_match_ = partial_match
        self.hwnd_ = None
        self.window_rect_ = None
        
        # 查找窗口
        self.findWindow()
        
    def findWindow(self) -> bool:
        """
        查找目标窗口
        
        Returns:
            是否找到窗口
        """
        if self.process_name_:
            self.hwnd_ = self._findWindowByProcess(self.process_name_)
        elif self.window_title_:
            self.hwnd_ = self._findWindowByTitle(self.window_title_, self.partial_match_)
        else:
            raise ValueError("Must provide either process_name or window_title")
        
        if self.hwnd_:
            self.updateWindowRect()
            logger.info(f"✓ 找到目标窗口: hwnd={self.hwnd_}")
            logger.info(f"  - 窗口标题: {win32gui.GetWindowText(self.hwnd_)}")
            logger.info(f"  - 窗口位置: {self.window_rect_}")
            logger.info(f"  - 窗口大小: {self.GetWidth()}x{self.GetHeight()}")
            return True
        else:
            logger.error("✗ 未找到目标窗口")
            if self.process_name_:
                logger.error(f"  - 进程名称: {self.process_name_}")
            if self.window_title_:
                logger.error(f"  - 窗口标题: {self.window_title_}")
            logger.error("\n请确保：")
            logger.error("  1. 游戏已启动")
            logger.error("  2. 进程名称或窗口标题正确")
            logger.error("  3. 游戏窗口未最小化")
            return False
    
    def _findWindowByProcess(self, process_name: str) -> Optional[int]:
        """通过进程名称查找窗口"""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    import psutil
                    proc = psutil.Process(pid)
                    if proc.name().lower() == process_name.lower():
                        windows.append(hwnd)
                except:
                    pass
            return True
        
        windows = []
        try:
            win32gui.EnumWindows(callback, windows)
            
            if windows:
                # 如果找到多个，选择第一个可见的主窗口
                for hwnd in windows:
                    if win32gui.GetWindowText(hwnd):  # 有标题的才是主窗口
                        return hwnd
                return windows[0]
        except Exception as e:
            logger.error(f"查找窗口失败: {e}")
        
        return None
    
    def _findWindowByTitle(self, title: str, partial: bool = True) -> Optional[int]:
        """通过窗口标题查找窗口"""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title:
                    if partial:
                        if title.lower() in window_title.lower():
                            windows.append((hwnd, window_title))
                    else:
                        if title == window_title:
                            windows.append((hwnd, window_title))
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if windows:
            # 如果找到多个，打印所有匹配的窗口
            if len(windows) > 1:
                logger.info(f"找到 {len(windows)} 个匹配的窗口：")
                for hwnd, wtitle in windows:
                    logger.info(f"  - {wtitle} (hwnd={hwnd})")
                logger.info("使用第一个窗口")
            
            return windows[0][0]
        
        return None
    
    def updateWindowRect(self):
        """更新窗口位置和大小"""
        if self.hwnd_:
            try:
                self.window_rect_ = win32gui.GetWindowRect(self.hwnd_)
            except:
                logger.warning("无法获取窗口位置，窗口可能已关闭")
                self.hwnd_ = None
                self.window_rect_ = None
    
    def Capture(self) -> Optional[np.ndarray]:
        """
        捕获窗口画面
        
        Returns:
            RGB 格式的 numpy 数组 (height, width, 3)
            如果失败返回 None
        """
        if not self.hwnd_:
            # 尝试重新查找窗口
            if not self.findWindow():
                return None
        
        # 更新窗口位置（窗口可能移动了）
        self.updateWindowRect()
        
        if not self.window_rect_:
            return None
        
        try:
            # 获取窗口设备上下文
            hwndDC = win32gui.GetWindowDC(self.hwnd_)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # 创建位图对象
            left, top, right, bottom = self.window_rect_
            width = right - left
            height = bottom - top
            
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # 捕获窗口内容
            result = saveDC.BitBlt(
                (0, 0),
                (width, height),
                mfcDC,
                (0, 0),
                win32con.SRCCOPY
            )
            
            # 转换为 numpy 数组
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img = img.reshape((height, width, 4))  # BGRA
            img = img[:, :, :3]  # 去掉 Alpha 通道
            img = np.ascontiguousarray(img[:, :, ::-1])  # BGR -> RGB
            
            # 清理资源
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd_, hwndDC)
            
            return img
            
        except Exception as e:
            logger.error(f"捕获窗口失败: {e}")
            # 窗口可能已关闭
            self.hwnd_ = None
            return None
    
    def GetWidth(self) -> int:
        """获取窗口宽度"""
        if self.window_rect_:
            return self.window_rect_[2] - self.window_rect_[0]
        return 0
    
    def GetHeight(self) -> int:
        """获取窗口高度"""
        if self.window_rect_:
            return self.window_rect_[3] - self.window_rect_[1]
        return 0
    
    def GetWindowRect(self) -> Optional[Tuple[int, int, int, int]]:
        """获取窗口矩形 (left, top, right, bottom)"""
        return self.window_rect_
    
    def IsValid(self) -> bool:
        """检查窗口句柄是否有效"""
        if not self.hwnd_:
            return False
        
        try:
            return win32gui.IsWindow(self.hwnd_) and win32gui.IsWindowVisible(self.hwnd_)
        except:
            return False


def listAllWindows():
    """列出所有可见窗口（调试用）"""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    import psutil
                    proc = psutil.Process(pid)
                    windows.append({
                        'hwnd': hwnd,
                        'title': title,
                        'process': proc.name(),
                        'pid': pid
                    })
                except:
                    windows.append({
                        'hwnd': hwnd,
                        'title': title,
                        'process': 'Unknown',
                        'pid': 0
                    })
        return True
    
    windows = []
    if WIN32_AVAILABLE:
        win32gui.EnumWindows(callback, windows)
    
    return windows


if __name__ == "__main__":
    """测试窗口捕获"""
    logger.info("=" * 80)
    logger.info("窗口捕获工具 - 测试模式")
    logger.info("=" * 80)
    
    # 列出所有窗口
    logger.info("\n所有可见窗口：")
    windows = listAllWindows()
    for w in windows[:20]:  # 只显示前 20 个
        logger.info(f"  - {w['title'][:50]:50s} | {w['process']:30s} | PID: {w['pid']}")
    
    if len(windows) > 20:
        logger.info(f"  ... 还有 {len(windows) - 20} 个窗口")
    
    # 测试捕获
    logger.info("\n请输入要捕获的窗口标题关键字（如 'chrome', 'notepad' 等）：")
    keyword = input("> ").strip()
    
    if keyword:
        try:
            wc = WindowCapture(window_title=keyword, partial_match=True)
            
            if wc.IsValid():
                logger.info("\n开始捕获...")
                import time
                for i in range(3):
                    frame = wc.Capture()
                    if frame is not None:
                        logger.info(f"✓ 捕获成功 {i+1}/3: shape={frame.shape}")
                        
                        # 保存第一帧
                        if i == 0:
                            import cv2
                            cv2.imwrite("test_capture.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            logger.info("  - 已保存到 test_capture.png")
                    else:
                        logger.error(f"✗ 捕获失败 {i+1}/3")
                    
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"测试失败: {e}")
            import traceback
            traceback.print_exc()

