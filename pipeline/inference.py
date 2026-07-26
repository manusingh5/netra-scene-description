# pipeline/inference.py
"""
NETRA Sequential Model Inference
Per-frame sequential processing: Florence-2 → YOLOv8 → PaddleOCR
Memory-efficient: loads models lazily, cleans up after each frame
"""

import gc
import time
from typing import List, Dict
from pipeline.extractor import VideoExtractor, FrameInfo

class SequentialInference:
    """
    Runs perception models sequentially on each frame
    Models loaded lazily on first use
    """

    def __init__(self):
        self._florence = None
        self._yolo = None
        self._ocr = None
        self._spatial = None

    # --- Lazy Model Loaders ---

    @property
    def florence(self):
        if self._florence is None:
            print("[INFERENCE] Loading Florence-2...")
            from models.florence_model import FlorenceModel
            self._florence = FlorenceModel()
            print("[INFERENCE] Florence-2 loaded ✅")
        return self._florence

    @property
    def yolo(self):
        if self._yolo is None:
            print("[INFERENCE] Loading YOLOv8-nano...")
            from models.yolo_model import YOLOModel
            self._yolo = YOLOModel()
            print("[INFERENCE] YOLOv8-nano loaded ✅")
        return self._yolo

    @property
    def ocr(self):
        if self._ocr is None:
            print("[INFERENCE] Loading PaddleOCR...")
            from models.ocr_model import OCRModel
            self._ocr = OCRModel()
            print("[INFERENCE] PaddleOCR loaded ✅")
        return self._ocr

    @property
    def spatial(self):
        if self._spatial is None:
            from utils.spatial_grid import SpatialGrid
            self._spatial = SpatialGrid()
        return self._spatial

    # --- Main Inference Loop ---

    def run(self, video_path: str, fps: int = 1) -> List[Dict]:
        """
        Run all perception models on each frame
        Returns list of frame descriptors
        """
        extractor = VideoExtractor(fps=fps)
        descriptors = []

        print("[INFERENCE] Starting sequential inference...")

        for frame_info in extractor.extract_frames(video_path):
            frame_start = time.time()

            # --- Step 1: Florence-2 Caption ---
            try:
             # Florence-2 सिर्फ हर 3rd frame पर चलाओ (speed बचाने के लिए)
               if frame_info.frame_id % 3 == 0:
                caption: str = self.florence.caption(frame_info.frame)
               else:
                caption = "[Skipped for speed]"
            except Exception as e:
                print(f"[INFERENCE] Florence-2 failed on frame {frame_info.frame_id}: {e}")
                caption = "[Description unavailable]"

            # --- Step 2: YOLOv8 Object Detection ---
            try:
                detections = self.yolo.detect(frame_info.frame)
            except Exception as e:
                print(f"[INFERENCE] YOLOv8 failed on frame {frame_info.frame_id}: {e}")
                detections = []

            # --- Step 3: PaddleOCR Text Extraction ---
            try:
                texts = self.ocr.extract_text(frame_info.frame)
            except Exception as e:
                print(f"[INFERENCE] OCR failed on frame {frame_info.frame_id}: {e}")
                texts = []

            # --- Step 4: Spatial Grid Mapping ---
            try:
                spatial_map = self.spatial.map_objects(detections, frame_info.frame.shape)
            except Exception as e:
                spatial_map = {}

            # --- Build Descriptor ---
            descriptor = {
                "frame_id": frame_info.frame_id,
                "timestamp": frame_info.timestamp_sec,
                "caption": caption,
                "objects": detections,
                "text_detected": texts,
                "spatial_map": spatial_map
            }

            descriptors.append(descriptor)

            elapsed = time.time() - frame_start
            print(f"[INFERENCE] Frame {frame_info.frame_id} done in {elapsed:.1f}s "
                  f"(caption={len(caption)} chars, objects={len(detections)}, texts={len(texts)})")

            # --- Memory Cleanup ---
            del frame_info
            gc.collect()

        print(f"[INFERENCE] Done! Total descriptors: {len(descriptors)}")
        return descriptors

    # --- Unload Perception Models (free RAM for LLM) ---

    def unload_perception_models(self):
        """Free memory occupied by perception models"""
        print("[INFERENCE] Unloading perception models...")

        if self._florence:
            del self._florence
            self._florence = None

        if self._yolo:
            del self._yolo
            self._yolo = None

        if self._ocr:
            del self._ocr
            self._ocr = None

        gc.collect()
        print("[INFERENCE] Perception models unloaded ✅ (RAM freed for LLM)")