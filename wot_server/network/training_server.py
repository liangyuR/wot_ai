"""
Linux 训练服务器 - 负责模型训练和推理
"""

import socket
import pickle
import numpy as np
import cv2
import threading
from typing import Optional, Dict, Any
from queue import Queue
from loguru import logger
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents import WotPpoAgent, LoadConfig


class TrainingServer:
    """
    Linux 端训练服务器
    
    功能：
    1. 接收 Windows 客户端发送的游戏画面
    2. 使用 AI 模型进行推理
    3. 发送动作指令回客户端
    4. 在后台进行模型训练
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9999,
        model_path: Optional[str] = None,
        config_path: str = "configs/ppo_config.yaml"
    ):
        """
        初始化训练服务器
        
        Args:
            host: 监听地址
            port: 监听端口
            model_path: 模型路径（如果为 None，使用随机策略）
            config_path: 配置文件路径
        """
        self.host_ = host
        self.port_ = port
        self.model_path_ = model_path
        self.config_path_ = config_path
        
        # 加载配置
        self.config_ = LoadConfig(config_path)
        
        # 加载模型
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.agent_ = WotPpoAgent(self.config_)
            self.agent_.load(model_path)
        else:
            logger.warning("No model provided, using random policy")
            self.agent_ = None
            
        # 网络
        self.server_socket_ = None
        self.client_socket_ = None
        self.running_ = False
        
        # 数据缓存
        self.frame_buffer_ = []
        self.action_buffer_ = []
        self.max_buffer_size_ = 10000
        
    def start(self):
        """启动服务器"""
        self.server_socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket_.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket_.bind((self.host_, self.port_))
        self.server_socket_.listen(1)
        
        logger.info(f"Server listening on {self.host_}:{self.port_}")
        self.running_ = True
        
        # 等待客户端连接
        logger.info("Waiting for game client to connect...")
        self.client_socket_, addr = self.server_socket_.accept()
        logger.info(f"Client connected from {addr}")
        
    def stop(self):
        """停止服务器"""
        self.running_ = False
        if self.client_socket_:
            self.client_socket_.close()
        if self.server_socket_:
            self.server_socket_.close()
        logger.info("Server stopped")
        
    def receiveFrame(self) -> Optional[Dict[str, Any]]:
        """
        接收游戏帧
        
        Returns:
            包含 frame, game_state 的字典
        """
        try:
            # 接收数据大小
            size_bytes = self.client_socket_.recv(4)
            if not size_bytes:
                return None
                
            size = int.from_bytes(size_bytes, byteorder='big')
            
            # 接收数据
            data = b''
            while len(data) < size:
                chunk = self.client_socket_.recv(min(size - len(data), 8192))
                if not chunk:
                    return None
                data += chunk
                
            # 反序列化
            frame_data = pickle.loads(data)
            
            # 解压图像
            frame_bytes = frame_data['frame']
            frame = cv2.imdecode(
                np.frombuffer(frame_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            return {
                'frame': frame,
                'game_state': frame_data['game_state'],
                'timestamp': frame_data['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Failed to receive frame: {e}")
            return None
            
    def sendAction(self, action: Dict[str, Any]):
        """
        发送动作到客户端
        
        Args:
            action: 动作字典
        """
        try:
            # 序列化
            serialized = pickle.dumps(action)
            size = len(serialized)
            
            # 发送大小和数据
            self.client_socket_.sendall(size.to_bytes(4, byteorder='big'))
            self.client_socket_.sendall(serialized)
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            
    def preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理游戏帧
        
        Args:
            frame: 原始帧
            
        Returns:
            预处理后的帧
        """
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 缩放
        resized = cv2.resize(gray, (84, 84))
        return resized
        
    def predictAction(self, frame: np.ndarray, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测动作
        
        Args:
            frame: 游戏画面
            game_state: 游戏状态
            
        Returns:
            动作字典
        """
        if self.agent_ is None:
            # 随机策略
            import random
            return {
                'move': random.choice(['none', 'forward', 'backward']),
                'turn': random.choice(['none', 'left', 'right']),
                'turret': random.choice(['none', 'left', 'right']),
                'shoot': random.random() < 0.1
            }
        else:
            # 使用模型预测
            # TODO: 构建完整的观察空间
            obs = {
                'screen': self.preprocessFrame(frame),
                'health': np.array([game_state.get('health', 1.0)]),
                'ammo': np.array([game_state.get('ammo', 10.0) / 100.0]),
                'minimap': np.zeros((64, 64, 3), dtype=np.uint8)
            }
            
            action_id, _ = self.agent_.predict(obs, deterministic=True)
            
            # 将动作ID转换为动作字典
            return self.actionIdToDict(action_id)
            
    def actionIdToDict(self, action_id: int) -> Dict[str, Any]:
        """
        将动作ID转换为动作字典
        
        Args:
            action_id: 动作编号（0-7）
            
        Returns:
            动作字典
        """
        action_map = {
            0: {'move': 'forward', 'turn': 'none', 'turret': 'none', 'shoot': False},
            1: {'move': 'backward', 'turn': 'none', 'turret': 'none', 'shoot': False},
            2: {'move': 'none', 'turn': 'left', 'turret': 'none', 'shoot': False},
            3: {'move': 'none', 'turn': 'right', 'turret': 'none', 'shoot': False},
            4: {'move': 'none', 'turn': 'none', 'turret': 'left', 'shoot': False},
            5: {'move': 'none', 'turn': 'none', 'turret': 'right', 'shoot': False},
            6: {'move': 'none', 'turn': 'none', 'turret': 'none', 'shoot': True},
            7: {'move': 'none', 'turn': 'none', 'turret': 'none', 'shoot': False},
        }
        return action_map.get(action_id, action_map[7])
        
    def runLoop(self):
        """主循环"""
        logger.info("Starting inference loop...")
        
        frame_count = 0
        
        try:
            while self.running_:
                # 接收帧
                data = self.receiveFrame()
                if data is None:
                    logger.warning("Lost connection to client")
                    break
                    
                frame = data['frame']
                game_state = data['game_state']
                
                # 预测动作
                action = self.predictAction(frame, game_state)
                
                # 发送动作
                self.sendAction(action)
                
                # 缓存数据用于训练
                self.frame_buffer_.append(frame)
                self.action_buffer_.append(action)
                
                # 限制缓存大小
                if len(self.frame_buffer_) > self.max_buffer_size_:
                    self.frame_buffer_.pop(0)
                    self.action_buffer_.pop(0)
                    
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
                    
        except KeyboardInterrupt:
            logger.info("Loop interrupted by user")
        finally:
            self.running_ = False
            
    def saveCollectedData(self, output_path: str):
        """
        保存收集的数据
        
        Args:
            output_path: 输出路径
        """
        import pickle
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存帧和动作
        with open(output_path / "collected_data.pkl", "wb") as f:
            pickle.dump({
                'frames': self.frame_buffer_,
                'actions': self.action_buffer_
            }, f)
            
        logger.info(f"Saved {len(self.frame_buffer_)} frames to {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="World of Tanks Training Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9999, help="Server port")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--config", type=str, default="configs/ppo_config.yaml", help="Config path")
    parser.add_argument("--save-data", type=str, default=None, help="Save collected data")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("World of Tanks - Training Server (Linux)")
    logger.info("=" * 80)
    logger.info(f"Listening on {args.host}:{args.port}")
    logger.info(f"Model: {args.model or 'Random Policy'}")
    logger.info("=" * 80)
    
    # 创建服务器
    server = TrainingServer(
        host=args.host,
        port=args.port,
        model_path=args.model,
        config_path=args.config
    )
    
    try:
        # 启动服务器
        server.start()
        
        # 运行主循环
        server.runLoop()
        
    finally:
        # 保存数据（如果指定）
        if args.save_data:
            server.saveCollectedData(args.save_data)
            
        server.stop()


if __name__ == "__main__":
    main()

