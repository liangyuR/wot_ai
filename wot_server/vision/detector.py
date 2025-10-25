"""
Object detection module for World of Tanks
Detects enemy tanks, obstacles, etc.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import torch


class TankDetector:
    """
    Tank detection using YOLOv8
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = "cuda"
    ):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            device: Device to run inference on
        """
        self.confidence_threshold_ = confidence_threshold
        self.iou_threshold_ = iou_threshold
        self.device_ = device
        
        # Load YOLO model
        self.model_ = YOLO(model_path)
        self.model_.to(device)
        
        # Class names (will be customized after fine-tuning)
        self.class_names_ = [
            "light_tank",
            "medium_tank",
            "heavy_tank",
            "tank_destroyer",
            "artillery"
        ]
        
    def detect(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        Detect tanks in image
        
        Args:
            image: Input image (BGR format)
            visualize: Whether to visualize detections
            
        Returns:
            List of (class_id, confidence, bbox) tuples
            bbox format: (x1, y1, x2, y2)
        """
        # Run inference
        results = self.model_.predict(
            image,
            conf=self.confidence_threshold_,
            iou=self.iou_threshold_,
            device=self.device_,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    # Extract box data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    detections.append((class_id, confidence, (x1, y1, x2, y2)))
                    
                    # Visualize if requested
                    if visualize:
                        self.drawDetection(
                            image,
                            (x1, y1, x2, y2),
                            class_id,
                            confidence
                        )
        
        return detections
        
    def drawDetection(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        class_id: int,
        confidence: float
    ):
        """
        Draw detection on image
        
        Args:
            image: Image to draw on
            bbox: Bounding box (x1, y1, x2, y2)
            class_id: Class ID
            confidence: Detection confidence
        """
        x1, y1, x2, y2 = bbox
        
        # Colors for different classes
        colors = [
            (0, 255, 0),    # light_tank - green
            (255, 255, 0),  # medium_tank - cyan
            (0, 0, 255),    # heavy_tank - red
            (255, 0, 255),  # tank_destroyer - magenta
            (0, 165, 255)   # artillery - orange
        ]
        
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{self.class_names_[class_id]}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Background for text
        cv2.rectangle(
            image,
            (x1, y1 - label_size[1] - 4),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        # Text
        cv2.putText(
            image,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
    def getClosestTarget(
        self,
        detections: List[Tuple[int, float, Tuple[int, int, int, int]]],
        image_center: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Get closest target to crosshair (image center)
        
        Args:
            detections: List of detections
            image_center: Image center coordinates (cx, cy)
            
        Returns:
            Target center coordinates (x, y) or None
        """
        if not detections:
            return None
            
        cx, cy = image_center
        min_distance = float('inf')
        closest_target = None
        
        for _, _, (x1, y1, x2, y2) in detections:
            # Calculate center of bounding box
            target_cx = (x1 + x2) // 2
            target_cy = (y1 + y2) // 2
            
            # Calculate distance to crosshair
            distance = np.sqrt((target_cx - cx)**2 + (target_cy - cy)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_target = (target_cx, target_cy)
                
        return closest_target
        
    def calculateAimOffset(
        self,
        target_pos: Tuple[int, int],
        crosshair_pos: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Calculate aim offset from crosshair to target
        
        Args:
            target_pos: Target position (x, y)
            crosshair_pos: Crosshair position (x, y)
            
        Returns:
            Normalized aim offset (dx, dy) in range [-1, 1]
        """
        tx, ty = target_pos
        cx, cy = crosshair_pos
        
        # Calculate offset
        dx = tx - cx
        dy = ty - cy
        
        # Normalize (assuming 1920x1080 screen)
        dx_norm = np.clip(dx / 960.0, -1.0, 1.0)
        dy_norm = np.clip(dy / 540.0, -1.0, 1.0)
        
        return dx_norm, dy_norm


# Example usage
if __name__ == "__main__":
    # Test with sample image
    detector = TankDetector()
    
    # Load test image
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Detect
    detections = detector.detect(test_image, visualize=True)
    
    print(f"Found {len(detections)} tanks")
    
    # Show result
    cv2.imshow("Detections", test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

