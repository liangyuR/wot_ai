from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

def main():
    weights_path = "yolo11m-seg.pt"
    model = YOLO(weights_path)

    image_path = Path("test.png")
    
    # 3. 推理
    results = model(str(image_path))  # 对 seg 模型，这样调用就会做分割推理
    res = results[0]

    # 4. 打印基本信息
    print("=== YOLO11m-seg 推理结果 ===")
    print("图片尺寸:", res.orig_shape)  # (h, w)

    boxes = res.boxes  # 边框
    masks = res.masks  # 分割掩码（可能为 None，如果没有预测到）

    print(f"检测到目标数量: {len(boxes) if boxes is not None else 0}")

    if boxes is not None:
        for i, b in enumerate(boxes):
            xyxy = b.xyxy[0].tolist()      # [x1, y1, x2, y2]
            cls_id = int(b.cls.item())     # 类别 id
            score = float(b.conf.item())   # 置信度
            print(f"[{i}] cls={cls_id}, score={score:.3f}, box={xyxy}")

    if masks is not None:
        print("存在分割掩码，mask 数量:", masks.data.shape[0])
    else:
        print("没有分割掩码（可能是没有检测到目标）")

    # Read the original image
    image = cv2.imread(str(image_path))

    if masks is not None:
        # Get masks data as numpy array (N, H, W)
        mask_data = masks.data.cpu().numpy()

        # Create an overlay to visualize all masks with random colors
        overlay = image.copy()
        for i in range(mask_data.shape[0]):
            mask = mask_data[i]
            color = tuple(int(x) for x in np.random.choice(range(128, 256), size=3))
            # Create a color mask and blend it
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = (mask * color[c]).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        cv2.imshow("image with masks", overlay)
        # Additionally, show the combined mask as grayscale
        combined_mask = (np.sum(mask_data, axis=0) > 0).astype(np.uint8) * 255
        cv2.imshow("combined masks", combined_mask)
    else:
        # No mask, just show image
        cv2.imshow("image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()