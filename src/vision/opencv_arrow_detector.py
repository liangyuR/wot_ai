import cv2
import numpy as np
import math
from typing import Tuple, Optional
from loguru import logger

class OpencvArrowDetector:
    """
    基于 OpenCV 的箭头检测器
    通过颜色分割提取箭头区域，利用 PCA 计算主轴，结合几何特征确定朝向。
    """
    
    def __init__(self):
        # 默认针对白色箭头配置 HSV 阈值
        # 根据实际箭头颜色可能需要调整
        self.lower_white = np.array([0, 0, 200])   # S低，V高
        self.upper_white = np.array([180, 50, 255])

    def detect(self, image: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """
        检测箭头中心和朝向
        
        Args:
            image: BGR 图像
            
        Returns:
            (center, angle): 
                center: (x, y) 元组
                angle: 角度（度），0度为向右，顺时针增加（或根据坐标系定义）
        """
        if image is None or image.size == 0:
            return None, None

        # 1. 预处理 & 颜色分割
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # 稍微膨胀一点以连接断裂区域
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # 2. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None

        # 找最大面积的轮廓
        c = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(c) < 20: # 忽略过小的噪点
            return None, None

        # 3. 计算重心 (Centroid)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None, None
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center = (cx, cy)

        # 4. 计算朝向
        # 使用 PCA (主成分分析) 计算主轴方向
        # 构造数据点矩阵 (N, 2)
        pts = c.reshape(-1, 2).astype(np.float64)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
        
        # 主特征向量 (对应最大特征值)
        # eigenvectors[0] 是最大特征值对应的向量
        vx, vy = eigenvectors[0]
        
        # PCA 算出的向量只有轴向，存在 180 度模糊
        # 策略：对于普通箭头，尖端通常是距离重心最远的点
        
        max_dist_sq = 0
        farthest_pt = None
        
        for pt in pts:
            px, py = pt
            dist_sq = (px - cx)**2 + (py - cy)**2
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                farthest_pt = (px, py)
        
        if farthest_pt is None:
            return center, None
            
        # 计算 重心 -> 最远点 的向量
        tip_vec_x = farthest_pt[0] - cx
        tip_vec_y = farthest_pt[1] - cy
        
        # 计算该向量与 PCA 主轴的点积
        dot_prod = tip_vec_x * vx + tip_vec_y * vy
        
        # 如果点积为负，说明 PCA 向量反了，翻转它
        if dot_prod < 0:
            vx, vy = -vx, -vy
            
        # 计算最终角度 (弧度转度)
        # 使用 atan2(y, x)，范围 [-180, 180]
        angle_rad = math.atan2(vy, vx)
        angle_deg = math.degrees(angle_rad)
        
        # 归一化到 [0, 360) 或保持 [-180, 180]
        # 这里保持与 atan2 一致，或者转为正数
        angle_deg = angle_deg % 360.0
        
        return center, angle_deg

    def debug_visualize(self, image: np.ndarray) -> np.ndarray:
        """
        调试可视化：在图像上绘制检测到的中心和方向
        """
        if image is None:
            return None
            
        vis_img = image.copy()
        center, angle = self.detect(image)
        
        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            
            # 绘制中心
            cv2.circle(vis_img, (cx, cy), 4, (0, 0, 255), -1)
            
            if angle is not None:
                # 绘制方向箭头
                length = 30
                angle_rad = math.radians(angle)
                end_x = int(cx + length * math.cos(angle_rad))
                end_y = int(cy + length * math.sin(angle_rad))
                
                cv2.arrowedLine(vis_img, (cx, cy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
                
                # 显示角度文字
                cv2.putText(vis_img, f"{angle:.1f} deg", (cx + 10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
        return vis_img

if __name__ == "__main__":
    import sys
    import os
    
    # 默认测试目录，可以通过命令行参数覆盖
    test_dir = "C:/data/20251203/images/"
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        # 如果目录不存在，回退到简单的单个测试示例（可选）
        # 这里直接退出提示用户
        sys.exit(1)

    detector = OpencvArrowDetector()
    
    # 支持的图片扩展名
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    
    # 获取并排序文件列表
    files = sorted([f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    if not files:
        print(f"在 {test_dir} 中未找到图片文件")
        sys.exit(0)
        
    print(f"找到 {len(files)} 张图片，按任意键查看下一张，按 ESC 退出。")
    
    for filename in files:
        file_path = os.path.join(test_dir, filename)
        test_img = cv2.imread(file_path)
        
        if test_img is None:
            print(f"无法读取图片: {filename}")
            continue
            
        center, angle = detector.detect(test_img)
        print(f"[{filename}] Detected: Center={center}, Angle={angle}")
        
        vis = detector.debug_visualize(test_img)
        
        cv2.imshow("Result", vis)
        key = cv2.waitKey(0)
        
        if key == 27: # ESC 键退出
            break

    cv2.destroyAllWindows()

