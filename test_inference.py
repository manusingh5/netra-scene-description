# test_inference.py
"""Test: Run sequential inference on a video"""
from pipeline.inference import SequentialInference
import glob

videos = glob.glob("uploads/*.mp4")

if not videos:
    print("❌ uploads/ में कोई mp4 नहीं मिला")
else:
    video = videos[0]
    print(f"Testing with: {video}\n")

    inference = SequentialInference()
    descriptors = inference.run(video, fps=1)

    print(f"\n--- Results ---")
    print(f"Total descriptors: {len(descriptors)}")
    
    if descriptors:
        print(f"\nFirst descriptor:")
        print(f"  Frame ID: {descriptors[0]['frame_id']}")
        print(f"  Timestamp: {descriptors[0]['timestamp']}s")
        print(f"  Caption: {descriptors[0]['caption']}")
        print(f"  Objects: {descriptors[0]['objects']}")
        print(f"  Text: {descriptors[0]['text_detected']}")
        print(f"  Spatial: {descriptors[0]['spatial_map']}")
    
    print("\n✅ Inference working!")