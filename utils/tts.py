# utils/tts.py
"""
NETRA Text-to-Speech (TTS) Module
Supports multiple TTS engines:
  - Pyttsx3 (offline, default for now)
  - Piper (higher quality, Phase 2)
"""

import os
from typing import Optional

class TTSEngine:
    """Abstract base class for TTS engines"""
    def generate(self, text: str, output_path: str) -> str:
        raise NotImplementedError

class Pyttsx3Engine(TTSEngine):
    """Offline TTS using pyttsx3 (no internet required)"""

    def __init__(self, rate: int = 150, volume: float = 0.9):
        self.rate = rate
        self.volume = volume
        self.engine = None

    def _ensure_engine(self):
        """Lazy load pyttsx3 engine"""
        if self.engine is None:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Try to set English voice
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'english' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

    def generate(self, text: str, output_path: str) -> str:
        """Generate audio file from text"""
        if not text or not text.strip():
            print("[TTS] Empty text, skipping generation")
            return None

        self._ensure_engine()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        print(f"[TTS] Generating audio: {text[:50]}...")
        
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
        
        print(f"[TTS] Audio saved to: {output_path}")
        return output_path

    def unload(self):
        """Free engine resources"""
        if self.engine:
            del self.engine
            self.engine = None
            print("[TTS] Engine unloaded")


class PiperEngine(TTSEngine):
    """High-quality offline TTS using Piper (better voice quality)"""

    def __init__(self, model_path: str = "piper_models/en_US-libritts-high"):
        self.model_path = model_path
        self.synthesizer = None

    def _ensure_synthesizer(self):
        if self.synthesizer is None:
            try:
                from piper import Synthesizer
                self.synthesizer = Synthesizer(self.model_path)
                print("[TTS] Piper synthesizer loaded ✅")
            except ImportError:
                print("[TTS] Piper not installed. Install with: pip install piper-tts")
                raise

    def generate(self, text: str, output_path: str) -> str:
        """Generate audio using Piper"""
        if not text or not text.strip():
            return None

        self._ensure_synthesizer()
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        print(f"[TTS] Piper: Generating audio: {text[:50]}...")
        
        audio_samples = self.synthesizer.synthesize(text)
        sample_rate = 22050
        
        import wave
        import struct
        
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            for sample in audio_samples:
                value = int(sample * 32767)
                value = max(-32768, min(32767, value))
                wav_file.writeframes(struct.pack('<h', value))
        
        print(f"[TTS] Piper audio saved to: {output_path}")
        return output_path

    def unload(self):
        if self.synthesizer:
            del self.synthesizer
            self.synthesizer = None
            print("[TTS] Piper unloaded")


# Default factory
_default_engine: Optional[TTSEngine] = None

def get_tts_engine(engine_type: str = "pyttsx3") -> TTSEngine:
    """Get TTS engine instance (singleton pattern)"""
    global _default_engine
    
    if _default_engine is None:
        if engine_type == "pyttsx3":
            _default_engine = Pyttsx3Engine()
        elif engine_type == "piper":
            _default_engine = PiperEngine()
        else:
            raise ValueError(f"Unknown TTS engine: {engine_type}")
    
    return _default_engine

def generate_audio(text: str, output_path: str, engine_type: str = "pyttsx3") -> str:
    """Convenience function to generate audio from text"""
    engine = get_tts_engine(engine_type)
    return engine.generate(text, output_path)

def unload_tts():
    """Unload TTS engine (free memory)"""
    global _default_engine
    if _default_engine:
        _default_engine.unload()
        _default_engine = None