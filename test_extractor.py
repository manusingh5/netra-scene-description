# test_extractor.py
"""Test: Extract frames from video at 1 FPS"""
from pipeline.extractor import VideoExtractor

extractor = VideoExtractor(fps=1)

# uploads/ में जो भी वीडियो है वो use करो
import glob
videos = glob.glob("uploads/*.mp4")

if not videos:
    print("❌ uploads/ में कोई mp4 नहीं मिला")
else:
    video = videos[0]
    print(f"Testing with: {video}")
    
    # Get video info
    info = extractor.get_video_info(video)
    print(f"Duration: {info['duration_sec']:.1f}s, Resolution: {info['width']}x{info['height']}")
    
    # Extract frames
    count = 0
    for frame_info in extractor.extract_frames(video):
        print(f"Frame {frame_info.frame_id}: t={frame_info.timestamp_sec}s, shape={frame_info.frame.shape}")
        count += 1
    
    print(f"\n✅ Extractor working! Extracted {count} frames")