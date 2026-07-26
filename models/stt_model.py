# models/stt_model.py

class STTModel:
    def __init__(self):
        self.model = None
        print("[STT] Stub initialized (placeholder)")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            return ""
        return ""

    def unload(self):
        if self.model:
            del self.model
            self.model = None