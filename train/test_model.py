#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克世界模型测试脚本
用于测试训练好的模仿学习模型
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import argparse
import logging

# 导入训练脚本中的模型和配置
from train_imitation import (
    TankImitationNet, TankImitationDataset,
    ACTION_DIM, SEQUENCE_LENGTH, TARGET_W, TARGET_H,
    ACTION_KEYS, MOUSE_KEYS, SPECIAL_KEYS, SCREEN_W, SCREEN_H
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 动态计算默认路径
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
DEFAULT_DATA_ROOT = str(_project_root / "data_collection" / "data" / "recordings")
DEFAULT_MODEL_PATH = str(_script_dir / "tank_imitation_model.pth")


class ModelTester:
    """模型测试器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        model = TankImitationNet(
            action_dim=ACTION_DIM,
            sequence_length=SEQUENCE_LENGTH
        )
        
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=self.device)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        logger.info(f"模型已加载: {model_path}")
        logger.info(f"动作维度: {ACTION_DIM}, 序列长度: {SEQUENCE_LENGTH}")
        return model
    
    def _get_transform(self):
        """获取图像变换"""
        from torchvision.transforms import InterpolationMode
        return transforms.Compose([
            transforms.Resize((TARGET_H, TARGET_W), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_image(self, image_path):
        """预测单张图像（需要构建序列输入）"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)  # [3, H, W]
        
        # 构建序列输入（重复当前帧以匹配序列长度）
        frame_sequence = image_tensor.unsqueeze(0).repeat(SEQUENCE_LENGTH, 1, 1, 1)  # [T, 3, H, W]
        frame_sequence = frame_sequence.unsqueeze(0).to(self.device)  # [1, T, 3, H, W]
        
        # 预测
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    prediction = self.model(frame_sequence)
            else:
                prediction = self.model(frame_sequence)
            prediction = prediction.cpu().numpy()[0]
        
        # 解析预测结果
        result = self._parse_prediction(prediction)
        return result
    
    def _parse_prediction(self, prediction):
        """解析预测结果"""
        # 分离按键和鼠标位移
        # 按键部分：前9维 [W, A, S, D, mouse_left, mouse_right, space, shift, t]
        key_pred = prediction[:9]
        # 鼠标位移部分：后2维 [dx_norm, dy_norm]
        mouse_pred = prediction[9:11]
        
        # 按键预测（使用sigmoid激活）
        key_probs = torch.sigmoid(torch.tensor(key_pred)).numpy()
        key_threshold = 0.5
        
        # 解析按键
        all_key_names = ACTION_KEYS + MOUSE_KEYS + SPECIAL_KEYS
        keys = []
        key_probs_dict = {}
        for i, (key, prob) in enumerate(zip(all_key_names, key_probs)):
            key_probs_dict[key] = float(prob)
            if prob > key_threshold:
                keys.append(key)
        
        # 解析鼠标位移（相对位移，需要归一化）
        max_step = 50.0
        dx = float(mouse_pred[0] * max_step)
        dy = float(mouse_pred[1] * max_step)
        
        # 分离各类按键
        mouse_left = "mouse_left" in keys
        mouse_right = "mouse_right" in keys
        movement_keys = [k for k in keys if k in ACTION_KEYS]
        special_keys = [k for k in keys if k in SPECIAL_KEYS]
        
        return {
            'movement_keys': movement_keys,
            'mouse_left': mouse_left,
            'mouse_right': mouse_right,
            'special_keys': special_keys,
            'mouse_delta': [dx, dy],
            'raw_prediction': prediction,
            'key_probabilities': key_probs_dict
        }
    
    def test_on_dataset(self, data_root, num_samples=10):
        """在数据集上测试模型"""
        dataset = TankImitationDataset(data_root, sequence_length=SEQUENCE_LENGTH)
        
        logger.info(f"在数据集上测试 {num_samples} 个样本")
        
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(min(num_samples, len(dataset))):
            # dataset[i] 返回 (frame_sequence [T, 3, H, W], action_vector_tensor)
            frame_sequence, target_action = dataset[i]
            # 从底层 samples 取出原始动作字典
            raw_action = dataset.samples[i][1]
            
            # 预测（需要添加batch维度）
            frame_batch = frame_sequence.unsqueeze(0).to(self.device)  # [1, T, 3, H, W]
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        prediction = self.model(frame_batch)
                else:
                    prediction = self.model(frame_batch)
                prediction = prediction.cpu().numpy()[0]
            
            # 解析预测和真实动作
            pred_result = self._parse_prediction(prediction)
            true_result = self._parse_true_action(raw_action)
            
            # 计算准确率
            accuracy = self._calculate_accuracy(pred_result, true_result)
            correct_predictions += accuracy
            total_predictions += 1
            
            logger.info(f"样本 {i+1}: 准确率 {accuracy:.2f}")
            logger.info(f"  预测: {pred_result['movement_keys']} "
                       f"鼠标delta({pred_result['mouse_delta']}) "
                       f"左键:{pred_result['mouse_left']} 右键:{pred_result['mouse_right']} "
                       f"特殊:{pred_result['special_keys']}")
            logger.info(f"  真实: {true_result['movement_keys']} "
                       f"鼠标delta({true_result['mouse_delta']}) "
                       f"左键:{true_result['mouse_left']} 右键:{true_result['mouse_right']} "
                       f"特殊:{true_result['special_keys']}")
        
        avg_accuracy = correct_predictions / total_predictions
        logger.info(f"平均准确率: {avg_accuracy:.2f}")
        return avg_accuracy
    
    def _parse_true_action(self, true_action):
        """解析真实动作"""
        keys = true_action.get('keys', [])
        mouse_pos = true_action.get('mouse_pos', [SCREEN_W//2, SCREEN_H//2])
        
        # 从samples中获取dx, dy（如果可用）
        movement_keys = [k for k in keys if k in ACTION_KEYS]
        mouse_left = "mouse_left" in keys
        mouse_right = "mouse_right" in keys
        special_keys = [k for k in keys if k in SPECIAL_KEYS]
        
        # 计算鼠标位移（相对值，需要从actions中提取）
        # 这里简化处理，实际应该从dataset的samples中获取dx, dy
        mouse_delta = [0.0, 0.0]  # 占位符
        
        return {
            'movement_keys': movement_keys,
            'mouse_left': mouse_left,
            'mouse_right': mouse_right,
            'special_keys': special_keys,
            'mouse_delta': mouse_delta,
            'mouse_position': mouse_pos
        }
    
    def _calculate_accuracy(self, pred, true):
        """计算预测准确率"""
        # 移动按键准确率
        pred_movement = set(pred['movement_keys'])
        true_movement = set(true['movement_keys'])
        movement_acc = len(pred_movement & true_movement) / max(len(pred_movement | true_movement), 1)
        
        # 鼠标按键准确率
        mouse_left_acc = 1.0 if pred['mouse_left'] == true['mouse_left'] else 0.0
        mouse_right_acc = 1.0 if pred['mouse_right'] == true['mouse_right'] else 0.0
        
        # 特殊按键准确率
        pred_special = set(pred['special_keys'])
        true_special = set(true['special_keys'])
        special_acc = len(pred_special & true_special) / max(len(pred_special | true_special), 1)
        
        # 鼠标位移准确率（允许一定误差，这里简化为忽略或使用较小的权重）
        # 由于是相对位移，难以直接比较，这里先忽略
        
        # 综合准确率（加权平均）
        overall_acc = (movement_acc * 0.4 + mouse_left_acc * 0.15 + 
                      mouse_right_acc * 0.15 + special_acc * 0.3)
        return overall_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试坦克世界模仿学习模型')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help='模型文件路径')
    parser.add_argument('--image', type=str, default=None,
                       help='单张图像测试路径')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                       help='数据集根目录')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='测试样本数量')
    
    args = parser.parse_args()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建测试器
    tester = ModelTester(args.model, device)
    
    if args.image:
        # 单张图像测试
        logger.info(f"测试单张图像: {args.image}")
        result = tester.predict_single_image(args.image)
        
        print("\n=== 预测结果 ===")
        print(f"移动按键: {result['movement_keys']}")
        print(f"鼠标左键: {result['mouse_left']}")
        print(f"鼠标右键: {result['mouse_right']}")
        print(f"鼠标位置: {result['mouse_position']}")
        print(f"按键概率: {result['key_probabilities']}")
        
    else:
        # 数据集测试
        if not Path(args.data_root).exists():
            logger.error(f"数据目录不存在: {args.data_root}")
            return
        
        accuracy = tester.test_on_dataset(args.data_root, args.num_samples)
        print(f"\n=== 测试结果 ===")
        print(f"平均准确率: {accuracy:.2f}")


if __name__ == "__main__":
    main()
