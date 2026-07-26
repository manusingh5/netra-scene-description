# test_yolo.py
from models.yolo_model import YOLOModel

yolo = YOLOModel()
results = yolo.detect(None)
print(f"Detections: {results}")
print("✅ YOLO stub working!")