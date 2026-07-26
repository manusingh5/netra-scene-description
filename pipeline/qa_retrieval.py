# pipeline/qa_retrieval.py
"""
NETRA Question Answering Retriever
Keyword-based retrieval of relevant descriptors + LLM answer generation
"""

from typing import List, Dict

class QARetriever:
    """Handles user questions about processed video"""

    def __init__(self):
        self._llm = None
        self.descriptors: List[Dict] = []
        self.narration: str = ""

    @property
    def llm(self):
        if self._llm is None:
            print("[QA] Loading Phi-3 Mini...")
            from models.llm_model import LLMModel
            self._llm = LLMModel()
            print("[QA] Phi-3 Mini loaded ✅")
        return self._llm

    def set_context(self, descriptors: List[Dict], narration: str = ""):
        self.descriptors = descriptors
        self.narration = narration

    def answer(self, question: str) -> str:
        if not self.descriptors:
            return "No video has been processed yet."

        relevant = self._retrieve_relevant(question)
        context = self._build_qa_context(relevant)
        prompt = self._build_qa_prompt(question, context)

        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            print(f"[QA] LLM failed: {e}")
            answer = self._fallback_answer(question, relevant)

        return answer

    def _retrieve_relevant(self, question: str) -> List[Dict]:
        question_lower = question.lower()
        scored = []

        for desc in self.descriptors:
            score = 0
            objects_text = " ".join(obj["name"].lower() for obj in desc.get("objects", []))
            caption_text = desc.get("caption", "").lower()

            for word in question_lower.split():
                if len(word) > 3:
                    if word in objects_text:
                        score += 5
                    if word in caption_text:
                        score += 3

            scored.append((score, desc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [desc for score, desc in scored[:5] if score > 0]

        if not top:
            top = self.descriptors[:3]

        return top

    def _build_qa_context(self, descriptors: List[Dict]) -> str:
        lines = []
        for desc in descriptors:
            time_range = desc.get("time_range", [0, 0])
            caption = desc.get("caption", "")
            objects = [obj["name"] for obj in desc.get("objects", [])]
            texts = desc.get("text_detected", [])

            line = f"At {time_range[0]:.0f}s: {caption}. "
            if objects:
                line += f"Objects: {', '.join(objects)}. "
            if texts:
                line += f"Text: {', '.join(texts)}."
            lines.append(line)

        return " ".join(lines)

    def _build_qa_prompt(self, question: str, context: str) -> str:
        return f"""You are NETRA, a helpful assistant for visually impaired users.
Answer the question based on the video scene context.

Context: {context}

Question: {question}

Answer:"""

    def _fallback_answer(self, question: str, descriptors: List[Dict]) -> str:
        q_lower = question.lower()
        all_objects = set()
        all_texts = set()

        for desc in descriptors:
            for obj in desc.get("objects", []):
                all_objects.add(obj["name"])
            for text in desc.get("text_detected", []):
                all_texts.add(text)

        if any(kw in q_lower for kw in ["object", "what", "see", "visible"]):
            if all_objects:
                return f"I can see: {', '.join(sorted(all_objects))}."
            return "No objects were detected."

        elif any(kw in q_lower for kw in ["text", "read", "write"]):
            if all_texts:
                return f"The text reads: {', '.join(sorted(all_texts))}."
            return "No readable text was found."

        else:
            if self.narration:
                return self.narration[:200] + "..."
            return "Based on the analysis, I can see various objects and activities."

    def unload_llm(self):
        if self._llm:
            del self._llm
            self._llm = None
            import gc
            gc.collect()
            print("[QA] LLM unloaded ✅")