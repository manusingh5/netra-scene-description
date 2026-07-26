# pipeline/summarization.py
"""
NETRA Scene Summarization
Uses Phi-3 Mini to generate natural language scene description
"""

from typing import List, Dict

class SceneSummarizer:
    """Converts structured descriptors into natural language narration"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            print("[SUMMARY] Loading Phi-3 Mini...")
            from models.llm_model import LLMModel
            self._llm = LLMModel()
            print("[SUMMARY] Phi-3 Mini loaded ✅")
        return self._llm

    def generate_narration(self, fused_descriptors: List[Dict]) -> str:
        """Generate a complete scene description from fused descriptors"""
        if not fused_descriptors:
            return "No scene information available."

        context = self._build_context(fused_descriptors)
        prompt = self._build_prompt(context)

        try:
            narration = self.llm.generate(prompt)
        except Exception as e:
            print(f"[SUMMARY] LLM failed: {e}")
            narration = self._fallback_narration(fused_descriptors)

        return narration

    def _build_context(self, descriptors: List[Dict]) -> str:
        lines = []
        for i, desc in enumerate(descriptors):
            time_range = desc.get("time_range", [0, 0])
            caption = desc.get("caption", "")
            objects = [obj["name"] for obj in desc.get("objects", [])]
            texts = desc.get("text_detected", [])

            line = f"Scene {i+1} (t={time_range[0]:.0f}s-{time_range[1]:.0f}s):\n"
            line += f"  Caption: {caption}\n"
            line += f"  Objects: {', '.join(objects) if objects else 'None'}\n"
            line += f"  Text: {', '.join(texts) if texts else 'None'}"
            lines.append(line)

        return "\n\n".join(lines)

    def _build_prompt(self, context: str) -> str:
        return f"""You are NETRA, a scene description assistant for visually impaired users.
Convert the following structured scene data into clear, natural spoken narration.
Be concise but informative.

{context}

Generate the narration:"""

    def _fallback_narration(self, descriptors: List[Dict]) -> str:
        parts = []
        for i, desc in enumerate(descriptors):
            objects = [obj["name"] for obj in desc.get("objects", [])]
            caption = desc.get("caption", "")
            texts = desc.get("text_detected", [])

            segment = f"Scene {i+1}. "
            if caption and caption != "[Description unavailable]":
                segment += caption + ". "
            if objects:
                segment += f"Objects visible: {', '.join(objects)}. "
            if texts:
                segment += f"Text reads: {', '.join(texts)}. "
            parts.append(segment)

        narration = " ".join(parts)
        return narration.strip() if narration else "No scene information available."

    def unload_llm(self):
        if self._llm:
            del self._llm
            self._llm = None
            import gc
            gc.collect()
            print("[SUMMARY] LLM unloaded ✅")