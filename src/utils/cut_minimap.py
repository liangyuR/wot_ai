#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量裁剪图像右下角小地图区域（支持递归 + 时间戳命名）
"""

import cv2
import os
import time
from pathlib import Path
import argparse


def crop_minimap(input_dir: str, output_dir: str, crop_width: int, crop_height: int) -> None:
    """
    批量裁剪图像右下角小地图部分
    
    Args:
        input_dir: 输入图像目录（支持递归搜索）
        output_dir: 输出图像目录
        crop_width: 裁剪区域宽度（像素）
        crop_height: 裁剪区域高度（像素）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 递归查找所有图片文件
    image_files = list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpg"))
    if not image_files:
        print(f"[警告] 未在 {input_dir} 或其子目录中找到图像文件。")
        return

    print(f"[信息] 找到 {len(image_files)} 张图像，开始处理...\n")

    for img_file in image_files:
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"[跳过] 无法读取图像: {img_file}")
            continue

        h, w, _ = image.shape
        x1 = max(0, w - crop_width)
        y1 = max(0, h - crop_height)
        cropped = image[y1:h, x1:w]

        # 获取文件创建时间戳（精确到秒）
        ctime = int(os.path.getctime(img_file))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ctime))

        out_name = f"{timestamp}_minimap.png"
        out_file = output_path / out_name

        cv2.imwrite(str(out_file), cropped)
        print(f"[成功] {img_file.name} -> {out_file.name}")

    print(f"\n✅ 全部完成！输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="批量裁剪图像右下角小地图区域（递归 + 时间戳命名）")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录（支持递归）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--width", type=int, required=True, help="小地图区域宽度（像素）")
    parser.add_argument("--height", type=int, required=True, help="小地图区域高度（像素）")

    args = parser.parse_args()
    crop_minimap(args.input_dir, args.output_dir, args.width, args.height)


if __name__ == "__main__":
    main()
