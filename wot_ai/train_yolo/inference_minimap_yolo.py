#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小地图检测模型推理脚本
加载训练好的模型并在新图像上测试检测效果
"""

import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO

# 统一导入机制
from loguru import logger
from wot_ai.utils.paths import setup_python_path
setup_python_path()

# 从配置文件加载类别定义（需要在 setup_python_path 之后导入）
from wot_ai.train_yolo.config_loader import LoadClassesFromConfig  # noqa: E402
from wot_ai.game_modules.navigation.core.minimap_detector import CLASS_MAPPING  # noqa: E402

# 颜色映射（BGR格式）- 根据类别数量自动生成
def GetColors(num_classes: int) -> dict:
    """根据类别数量生成颜色映射"""
    base_colors = [
        (0, 255, 255),    # 黄色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 红色
        (255, 0, 0),      # 蓝色
        (128, 128, 128),  # 灰色
        (255, 255, 0),    # 青色
        (255, 0, 255),    # 洋红
        (0, 255, 255),    # 黄色
    ]
    return {i: base_colors[i % len(base_colors)] for i in range(num_classes)}


def DrawDetections(
    image: np.ndarray,
    detections: List[dict],
    conf_threshold: float = 0.25,
    show_labels: bool = True,
    show_conf: bool = True,
    class_names: Optional[List[str]] = None,
    config_path: Optional[Path] = None
) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像（BGR格式）
        detections: 检测结果列表，每个元素包含 'class', 'confidence', 'bbox'
        conf_threshold: 置信度阈值
        show_labels: 是否显示类别标签
        show_conf: 是否显示置信度
        class_names: 类别名称列表，如果为 None 则从配置文件加载
        config_path: 配置文件路径，如果为 None 则使用默认路径
    
    Returns:
        绘制了检测结果的图像
    """
    if class_names is None:
        class_names = LoadClassesFromConfig(config_path)
    
    num_classes = len(class_names)
    colors = GetColors(num_classes)
    
    result = image.copy()
    
    for det in detections:
        cls_id = det['class']
        conf = det['confidence']
        bbox = det['bbox']  # [x1, y1, x2, y2]
        
        if conf < conf_threshold:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        color = colors.get(cls_id, (255, 255, 255))
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        
        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        if show_labels or show_conf:
            label_parts = []
            if show_labels:
                label_parts.append(class_name)
            if show_conf:
                label_parts.append(f"{conf:.2f}")
            
            label = " ".join(label_parts)
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制文本背景
            cv2.rectangle(
                result,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return result


def InferenceImage(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.03,
    iou_threshold: float = 0.5,
    output_path: Optional[str] = None,
    show_result: bool = True
) -> bool:
    """
    对单张图像进行推理
    
    Args:
        model_path: 模型文件路径
        image_path: 图像文件路径
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值（用于 NMS）
        output_path: 输出图像路径（可选）
        show_result: 是否显示结果
    
    Returns:
        是否成功
    """
    try:
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 从模型获取类别名称
        num_classes = len(model.names)
        class_names = [model.names[i] for i in range(num_classes)]
        logger.info(f"模型包含 {num_classes} 个类别: {class_names}")
        
        # 加载图像
        logger.info(f"加载图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return False
        
        # 推理
        logger.info("正在推理...")
        results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # 解析检测结果
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detections.append({
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })
        
        logger.info(f"检测到 {len(detections)} 个目标")
        
        # 打印每个检测结果（用于观察哪些类别的置信度上升最快）
        if detections:
            logger.info("检测结果详情:")
            for det in detections:
                class_name = class_names[det['class']] if det['class'] < len(class_names) else f"class_{det['class']}"
                logger.info(f"  {det} -> 类别: {class_name}, 置信度: {det['confidence']:.4f}, 边界框: {det['bbox']}")
                print(det)  # 直接打印到控制台
        
        # 绘制检测结果
        result_image = DrawDetections(image, detections, conf_threshold, class_names=class_names)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, result_image)
            logger.info(f"结果已保存: {output_path}")
        
        # 显示结果
        if show_result:
            cv2.imshow("Detection Result", result_image)
            logger.info("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        logger.error(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def InferenceDirectory(
    model_path: str,
    image_dir: str,
    conf_threshold: float = 0.03,
    iou_threshold: float = 0.5,
    output_dir: Optional[str] = None,
    show_result: bool = False
) -> bool:
    """
    对目录中的所有图像进行推理
    
    Args:
        model_path: 模型文件路径
        image_dir: 图像目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值（用于 NMS）
        output_dir: 输出目录（可选）
        show_result: 是否显示每张图像的结果
    
    Returns:
        是否成功
    """
    try:
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 从模型获取类别名称
        num_classes = len(model.names)
        class_names = [model.names[i] for i in range(num_classes)]
        logger.info(f"模型包含 {num_classes} 个类别: {class_names}")
        
        # 获取图像文件列表
        image_dir_path = Path(image_dir)
        image_files = list(image_dir_path.glob("*.png")) + list(image_dir_path.glob("*.jpg"))
        
        if len(image_files) == 0:
            logger.error(f"目录中没有图像文件: {image_dir}")
            return False
        
        logger.info(f"找到 {len(image_files)} 张图像")
        
        # 创建输出目录
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 处理每张图像
        success_count = 0
        for img_file in image_files:
            logger.info(f"处理: {img_file.name}")
            
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"无法读取图像: {img_file}")
                continue
            
            # 推理
            results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            # 解析检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detections.append({
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
            
            logger.info(f"  检测到 {len(detections)} 个目标")
            
            # 打印每个检测结果（用于观察哪些类别的置信度上升最快）
            if detections:
                logger.info(f"  [{img_file.name}] 检测结果详情:")
                for det in detections:
                    class_name = class_names[det['class']] if det['class'] < len(class_names) else f"class_{det['class']}"
                    logger.info(f"    {det} -> 类别: {class_name}, 置信度: {det['confidence']:.4f}, 边界框: {det['bbox']}")
                    print(f"[{img_file.name}] {det}")  # 直接打印到控制台
            
            # 绘制检测结果
            result_image = DrawDetections(image, detections, conf_threshold, class_names=class_names)
            
            # 保存结果
            if output_dir:
                output_path = output_dir_path / f"result_{img_file.name}"
                cv2.imwrite(str(output_path), result_image)
            
            # 显示结果
            if show_result:
                cv2.imshow(f"Detection: {img_file.name}", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            success_count += 1
        
        logger.info(f"处理完成: {success_count}/{len(image_files)} 张图像")
        return True
        
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def LoadTestConfig(config_path: Path) -> dict:
    """
    从配置文件加载测试配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        测试配置字典，如果不存在则返回空字典
    """
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}，将使用命令行参数")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        test_config = config.get('test', {})
        return test_config
    except Exception as e:
        logger.warning(f"加载测试配置失败: {e}，将使用命令行参数")
        return {}


def main():
    """主函数"""
    # 获取项目根目录
    from wot_ai.utils.paths import get_project_root
    project_root = get_project_root()
    config_path = project_root / "wot_ai" / "train_yolo" / "train_config.yaml"
    
    # 加载配置文件中的测试配置
    test_config = LoadTestConfig(config_path)
    
    # 从配置文件获取默认值
    default_model = test_config.get('model', None)
    default_image = test_config.get('image', None)
    default_image_dir = test_config.get('image_dir', None)
    default_output_dir = test_config.get('output_dir', None)
    default_conf = test_config.get('conf_threshold', 0.03)
    default_iou = test_config.get('iou_threshold', 0.5)
    default_show = test_config.get('show_result', True)
    
    parser = argparse.ArgumentParser(description="小地图检测模型推理")
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        required=default_model is None,
        help=f"模型文件路径（默认从配置文件读取: {default_model}）"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help=f"单张图像路径（默认从配置文件读取: {default_image}）"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=default_image_dir,
        help=f"图像目录路径（批量处理，默认从配置文件读取: {default_image_dir}）"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=default_conf,
        help=f"置信度阈值（默认从配置文件读取: {default_conf}）"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=default_iou,
        help=f"IoU 阈值（默认从配置文件读取: {default_iou}）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output_dir,
        help=f"输出路径（图像文件或目录，默认从配置文件读取: {default_output_dir}）"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="不显示结果窗口（覆盖配置文件中的 show_result）"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示结果窗口（覆盖配置文件中的 show_result）"
    )
    
    args = parser.parse_args()
    
    # 处理显示选项（命令行参数优先）
    if args.no_show:
        show_result = False
    elif args.show:
        show_result = True
    else:
        show_result = default_show
    
    # 确定使用单张图像还是目录
    use_image = args.image is not None
    use_dir = args.dir is not None
    
    if not use_image and not use_dir:
        parser.error("必须指定 --image 或 --dir，或在配置文件中设置 test.image_dir")
    
    if use_image and use_dir:
        parser.error("不能同时指定 --image 和 --dir")
    
    # 解析模型路径
    model_path = Path(args.model)
    if not model_path.is_absolute():
        # 尝试相对于项目根目录
        model_path = project_root / model_path
        # 如果还是不存在，尝试相对于训练输出目录
        if not model_path.exists():
            training_dir = project_root / "data" / "models" / "training" / "minimap"
            # 尝试在最近的实验目录中查找
            if training_dir.exists():
                exp_dirs = sorted(training_dir.glob("exp_*"), key=lambda x: x.stat().st_mtime, reverse=True)
                for exp_dir in exp_dirs:
                    potential_model = exp_dir / "weights" / args.model
                    if potential_model.exists():
                        model_path = potential_model
                        logger.info(f"在 {exp_dir.name} 中找到模型: {model_path}")
                        break
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        exit(1)
    
    success = False
    if use_image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        
        output_path = args.output
        if output_path:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = project_root / output_path
        
        success = InferenceImage(
            str(model_path),
            str(image_path),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            output_path=str(output_path) if output_path else None,
            show_result=show_result
        )
    else:
        image_dir = Path(args.dir)
        if not image_dir.is_absolute():
            image_dir = project_root / image_dir
        
        output_dir = args.output
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = project_root / output_dir
        
        success = InferenceDirectory(
            str(model_path),
            str(image_dir),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            output_dir=str(output_dir) if output_dir else None,
            show_result=show_result
        )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

