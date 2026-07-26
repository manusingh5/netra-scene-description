# models/yolo_model.py
"""
NETRA YOLOv8-nano Object Detection Wrapper
REAL MODEL — Using ultralytics YOLOv8-nano
"""

from typing import List, Dict
import cv2

class YOLOModel:
    """
    YOLOv8-nano object detection wrapper
    Real model loaded on initialization (~300MB)
    """

    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.25
        print("[YOLO] Loading YOLOv8-nano...")
        
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")  # Nano model (~300MB)
            print("[YOLO] YOLOv8-nano loaded successfully! ✅")
        except ImportError:
            print("[YOLO ERROR] ultralytics not found. Install with: pip install ultralytics")
        except Exception as e:
            print(f"[YOLO ERROR] Failed to load model: {e}")

    def detect(self, frame) -> List[Dict]:
        """
        Detect objects in a frame using YOLOv8-nano

        Args:
            frame: numpy array (BGR image) from OpenCV

        Returns:
            List of detected objects:
            [
                {
                    "name": "person",
                    "confidence": 0.95,
                    "bbox": [x1, y1, x2, y2]
                },
                ...
            ]
        """
        if self.model is None:
            print("[YOLO WARNING] Model not loaded. Returning empty detections.")
            return []

        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for r in results:
                for box in r.boxes:
                    # Extract object info
                    cls_id = int(box.cls[0])
                    obj_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    detections.append({
                        "name": obj_name,
                        "confidence": conf,
                        "bbox": xyxy
                    })

            return detections

        except Exception as e:
            print(f"[YOLO ERROR] Inference failed: {e}")
            return []

    def unload(self):
        """Free model memory"""
        if self.model:
            del self.model
            self.model = None
            print("[YOLO] Model unloaded")