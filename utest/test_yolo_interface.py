from ultralytics import YOLO
import cv2

model = YOLO("./best.pt")
print("Classes:", model.names)

image = cv2.imread("./0f4e31a8-frame_000100.png")
# 尝试 RGB 输入
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image, conf=0.05, verbose=True)
results[0].show()
