# test_tts.py
"""Test: Generate audio using TTS"""
from utils.tts import generate_audio
import os

# Sample narration text
narration = "A man is sitting at a desk with a laptop. There is a window behind him with daylight coming through. Objects visible include person, desk, chair, laptop, and window."

# Output path
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/test_narration.wav"

# Generate audio
result = generate_audio(narration, output_path, engine_type="pyttsx3")

if result:
    print(f"✅ TTS working! Audio saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Play it (Windows)
    import winsound
    print("Playing audio...")
    winsound.PlaySound(output_path, winsound.SND_FILENAME)
else:
    print("❌ TTS generation failed")