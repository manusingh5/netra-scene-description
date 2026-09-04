# models/stt_model.py
"""
Speech-to-Text Model (Whisper Tiny)
Converts audio recordings to text using librosa + whisper
"""

import whisper
import torch
import os
import numpy as np
import librosa

class STTModel:
    def __init__(self, model_name="tiny"):
        """Load Whisper model (CPU optimized)"""
        self.model_name = model_name
        self.model = None
        self.device = "cpu"
        print(f"[STT] Loading Whisper {model_name} model...")
        
    def load(self):
        """Load the model (lazy loading)"""
        if self.model is None:
            try:
                self.model = whisper.load_model(self.model_name, device=self.device)
                print(f"[STT] Whisper loaded successfully")
            except Exception as e:
                print(f"[STT] ERROR loading Whisper: {e}")
                raise
        
    def _load_audio_librosa(self, audio_path: str, target_sr=16000):
        """Load audio file using librosa (handles ffmpeg internally)"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return audio
        except Exception as e:
            print(f"[STT] Librosa load error: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text using librosa + whisper"""
        try:
            if self.model is None:
                self.load()
            
            # Convert to absolute path
            audio_path = os.path.abspath(audio_path)
            
            if not os.path.exists(audio_path):
                print(f"[STT] Audio file not found: {audio_path}")
                return ""
            
            # Load audio using librosa (handles ffmpeg)
            audio_array = self._load_audio_librosa(audio_path)
            
            # Whisper expects numpy array or file path
            result = self.model.transcribe(audio_array, language="english")
            text = result.get("text", "").strip()
            print(f"[STT] Transcription successful: '{text[:100]}...'")
            return text
            
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def unload(self):
        """Free memory"""
        if self.model:
            del self.model
            self.model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("[STT] Model unloaded")

# Singleton instance
_stt_instance = None

def get_stt_model():
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = STTModel()
    return _stt_instance