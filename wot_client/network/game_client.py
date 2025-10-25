"""
Windows 游戏客户端 - 负责游戏交互和数据采集
"""

import socket
import pickle
import numpy as np
import cv2
import time
import threading
from typing import Optional, Dict, Any
from queue import Queue
from loguru import logger
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from cpp_bindings import ScreenCapture, InputControl
    CPP_AVAILABLE = True
except ImportError:
    logger.warning("C++ bindings not available, using fallback")
    CPP_AVAILABLE = False
    import mss


class GameClient:
    """
    Windows 端游戏客户端
    
    功能：
    1. 捕获游戏画面
    2. 发送画面到 Linux 训练服务器
    3. 接收动作指令并执行
    """
    
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 9999,
        screen_width: int = 1920,
        screen_height: int = 1080
    ):
        """
        初始化游戏客户端
        
        Args:
            server_host: Linux 服务器地址
            server_port: 服务器端口
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self.server_host_ = server_host
        self.server_port_ = server_port
        self.screen_width_ = screen_width
        self.screen_height_ = screen_height
        
        # 网络连接
        self.socket_ = None
        self.connected_ = False
        
        # 初始化屏幕捕获
        if CPP_AVAILABLE:
            self.screen_capture_ = ScreenCapture(screen_width, screen_height)
            logger.info("Using C++ screen capture")
        else:
            self.sct_ = mss.mss()
            self.monitor_ = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
            logger.info("Using Python fallback screen capture")
            
        # 初始化输入控制
        if CPP_AVAILABLE:
            self.input_control_ = InputControl()
            logger.info("Using C++ input control")
        else:
            from pynput import keyboard, mouse
            self.keyboard_controller_ = keyboard.Controller()
            self.mouse_controller_ = mouse.Controller()
            logger.info("Using Python fallback input control")
            
        # 动作队列
        self.action_queue_ = Queue(maxsize=10)
        
        # 运行标志
        self.running_ = False
        
    def connect(self) -> bool:
        """连接到训练服务器"""
        try:
            self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_.connect((self.server_host_, self.server_port_))
            self.connected_ = True
            logger.info(f"Connected to server {self.server_host_}:{self.server_port_}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected_ = False
            return False
            
    def disconnect(self):
        """断开连接"""
        if self.socket_:
            self.socket_.close()
        self.connected_ = False
        logger.info("Disconnected from server")
        
    def captureScreen(self) -> np.ndarray:
        """捕获屏幕"""
        if CPP_AVAILABLE:
            buffer = self.screen_capture_.Capture()
            frame = np.frombuffer(buffer, dtype=np.uint8)
            frame = frame.reshape((self.screen_height_, self.screen_width_, 3))
        else:
            screenshot = self.sct_.grab(self.monitor_)
            frame = np.array(screenshot)[:, :, :3]
            
        return frame
        
    def sendFrame(self, frame: np.ndarray, game_state: Optional[Dict[str, Any]] = None):
        """
        发送游戏帧到服务器
        
        Args:
            frame: 游戏画面
            game_state: 游戏状态（血量、弹药等）
        """
        if not self.connected_:
            return
            
        try:
            # 压缩图像以减少网络传输
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # 打包数据
            data = {
                'frame': encoded.tobytes(),
                'shape': frame.shape,
                'game_state': game_state or {},
                'timestamp': time.time()
            }
            
            # 序列化并发送
            serialized = pickle.dumps(data)
            size = len(serialized)
            
            # 发送数据大小（4字节）
            self.socket_.sendall(size.to_bytes(4, byteorder='big'))
            # 发送数据
            self.socket_.sendall(serialized)
            
        except Exception as e:
            logger.error(f"Failed to send frame: {e}")
            self.connected_ = False
            
    def receiveAction(self) -> Optional[Dict[str, Any]]:
        """
        接收服务器发送的动作指令
        
        Returns:
            动作字典，例如 {'move': 'forward', 'turn': 'left', 'shoot': True}
        """
        if not self.connected_:
            return None
            
        try:
            # 接收数据大小
            size_bytes = self.socket_.recv(4)
            if not size_bytes:
                return None
                
            size = int.from_bytes(size_bytes, byteorder='big')
            
            # 接收数据
            data = b''
            while len(data) < size:
                chunk = self.socket_.recv(min(size - len(data), 4096))
                if not chunk:
                    return None
                data += chunk
                
            # 反序列化
            action = pickle.loads(data)
            return action
            
        except Exception as e:
            logger.error(f"Failed to receive action: {e}")
            return None
            
    def executeAction(self, action: Dict[str, Any]):
        """
        执行动作
        
        Args:
            action: 动作字典
        """
        if CPP_AVAILABLE:
            self._executeActionCpp(action)
        else:
            self._executeActionPython(action)
            
    def _executeActionCpp(self, action: Dict[str, Any]):
        """使用 C++ 模块执行动作"""
        # 移动
        move = action.get('move', 'none')
        if move == 'forward':
            self.input_control_.PressKey('w')
        elif move == 'backward':
            self.input_control_.PressKey('s')
        else:
            self.input_control_.ReleaseKey('w')
            self.input_control_.ReleaseKey('s')
            
        # 转向
        turn = action.get('turn', 'none')
        if turn == 'left':
            self.input_control_.PressKey('a')
        elif turn == 'right':
            self.input_control_.PressKey('d')
        else:
            self.input_control_.ReleaseKey('a')
            self.input_control_.ReleaseKey('d')
            
        # 炮塔
        turret = action.get('turret', 'none')
        if turret == 'left':
            self.input_control_.PressKey('q')
        elif turret == 'right':
            self.input_control_.PressKey('e')
        else:
            self.input_control_.ReleaseKey('q')
            self.input_control_.ReleaseKey('e')
            
        # 射击
        if action.get('shoot', False):
            self.input_control_.MouseClick()
            
    def _executeActionPython(self, action: Dict[str, Any]):
        """使用 Python 模块执行动作"""
        from pynput.keyboard import Key
        
        # 简化版本，使用 pynput
        move = action.get('move', 'none')
        if move == 'forward':
            self.keyboard_controller_.press('w')
        elif move == 'backward':
            self.keyboard_controller_.press('s')
        else:
            self.keyboard_controller_.release('w')
            self.keyboard_controller_.release('s')
            
    def runLoop(self, fps: int = 30):
        """
        主循环：捕获画面 -> 发送 -> 接收动作 -> 执行
        
        Args:
            fps: 运行帧率
        """
        self.running_ = True
        frame_interval = 1.0 / fps
        
        logger.info(f"Starting game loop at {fps} FPS")
        
        try:
            while self.running_:
                loop_start = time.time()
                
                # 1. 捕获画面
                frame = self.captureScreen()
                
                # 2. 提取游戏状态（TODO: 实现 OCR）
                game_state = {
                    'health': 1.0,  # Placeholder
                    'ammo': 10
                }
                
                # 3. 发送到服务器
                self.sendFrame(frame, game_state)
                
                # 4. 接收动作
                action = self.receiveAction()
                
                # 5. 执行动作
                if action:
                    self.executeAction(action)
                    
                # 维持帧率
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Loop interrupted by user")
        finally:
            self.running_ = False
            
    def stop(self):
        """停止运行"""
        self.running_ = False
        self.disconnect()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="World of Tanks Game Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=9999, help="Server port")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--width", type=int, default=1920, help="Screen width")
    parser.add_argument("--height", type=int, default=1080, help="Screen height")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("World of Tanks - Game Client (Windows)")
    logger.info("=" * 80)
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info(f"Screen: {args.width}x{args.height}")
    logger.info(f"FPS: {args.fps}")
    logger.info("=" * 80)
    
    # 创建客户端
    client = GameClient(
        server_host=args.host,
        server_port=args.port,
        screen_width=args.width,
        screen_height=args.height
    )
    
    # 连接服务器
    if not client.connect():
        logger.error("Failed to connect to server. Make sure the training server is running.")
        return
        
    try:
        # 运行主循环
        client.runLoop(fps=args.fps)
    finally:
        client.stop()


if __name__ == "__main__":
    main()

