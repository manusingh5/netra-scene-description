# pipeline/extractor.py
"""
NETRA Video Frame Extractor
Streams frames from video at 1 FPS using OpenCV
Memory-efficient: yields one frame at a time (no full array)
"""

import cv2
import os
from typing import Generator
from dataclasses import dataclass


@dataclass
class FrameInfo:
    """Metadata for each extracted frame"""
    frame_id: int
    timestamp_sec: float
    frame: any  # numpy ndarray (BGR)


class VideoExtractor:
    """
    Extract frames from video at 1 FPS (configurable)
    Uses generator pattern — one frame at a time, no memory bloat
    """

    def __init__(self, fps: float = 0.5, max_frames: int = 15):
        """
        Args:
            fps: Frames per second to extract (default 1 = every 1 sec)
            max_frames: Safety limit (300 = 5 min video at 1 FPS)
        """
        self.fps = fps
        self.max_frames = max_frames

    def extract_frames(self, video_path: str) -> Generator[FrameInfo, None, None]:
        """
        Stream frames from video — ONE AT A TIME
        Yields FrameInfo with frame_id, timestamp, and BGR image
        
        Usage:
            extractor = VideoExtractor(fps=1)
            for frame_info in extractor.extract_frames("video.mp4"):
                # Process frame_info.frame
                # After processing, frame gets garbage collected
                pass
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / video_fps if video_fps > 0 else 0

        print(f"[EXTRACTOR] Video: {video_path}")
        print(f"[EXTRACTOR] FPS: {video_fps}, Total frames: {total_frames}, Duration: {duration_sec:.1f}s")

        # Calculate frame interval
        frame_interval = int(video_fps / self.fps) if video_fps > 0 else 1
        if frame_interval < 1:
            frame_interval = 1

        frame_id = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only yield frames at the specified interval
            if frame_id % frame_interval == 0:
                timestamp = frame_id / video_fps if video_fps > 0 else frame_id
                yield FrameInfo(
                    frame_id=extracted_count,
                    timestamp_sec=round(timestamp, 2),
                    frame=frame
                )
                extracted_count += 1

                if extracted_count >= self.max_frames:
                    print(f"[EXTRACTOR] Max frame limit reached: {self.max_frames}")
                    break

            frame_id += 1

        cap.release()
        print(f"[EXTRACTOR] Extracted {extracted_count} frames at {self.fps} FPS")

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata without extracting frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        info = {
            "path": video_path,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_sec": 0
        }
        info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

        cap.release()
        return info