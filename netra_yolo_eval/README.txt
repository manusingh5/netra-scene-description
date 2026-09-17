NETRA YOLO evaluation annotations
=================================
These labels were manually estimated from the 7 supplied frames.

Policy:
- COCO class IDs are preserved (person=0, car=2, bus=5, traffic light=9, handbag=26, suitcase=28).
- Clear/recognizable instances were annotated.
- Very tiny or ambiguous distant objects were omitted consistently where practical.
- Preview images are included for visual review before treating the resulting mAP as a final reported metric.

Folder layout:
images/val/*.jpg
labels/val/*.txt
previews/*_preview.jpg
data.yaml

Run from this folder (or use an absolute path to data.yaml):
yolo detect val model=yolov8n.pt data=data.yaml imgsz=640
