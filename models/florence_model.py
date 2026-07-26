# models/florence_model.py
"""
NETRA Florence-2 Scene Captioning Wrapper
REAL MODEL — Using Hugging Face transformers Florence-2
"""

import os

# ============================================================
# Cache path SET KARNA — sabse pehle, BEFORE any other import
# Ye hi main problem thi — environment variables import ke baad
# set ho rahe the, isliye HuggingFace default path use kar raha tha
# ============================================================
cache_dir = os.path.join(os.path.expanduser('~'), 'hf_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir



import sys
import types
import importlib.util

# Proper fake flash_attn module with __spec__
_fake = types.ModuleType('flash_attn')
_fake.__spec__ = importlib.util.spec_from_loader('flash_attn', loader=None)
_fake.__path__ = []
_fake.is_flash_attn_2_available = lambda: False
sys.modules['flash_attn'] = _fake

_fake_sub1 = types.ModuleType('flash_attn.bert_padding')
_fake_sub1.__spec__ = importlib.util.spec_from_loader('flash_attn.bert_padding', loader=None)
sys.modules['flash_attn.bert_padding'] = _fake_sub1

_fake_sub2 = types.ModuleType('flash_attn.fused_bias')
_fake_sub2.__spec__ = importlib.util.spec_from_loader('flash_attn.fused_bias', loader=None)
sys.modules['flash_attn.fused_bias'] = _fake_sub2

# Ab transformers import karo
import torch
from typing import List, Dict


class FlorenceModel:
    def __init__(self):
        self.model = None
        self.processor = None
        print("[FLORENCE] Loading Florence-2...")

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM

            model_name = "microsoft/Florence-2-base"

            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            # CPU-safe settings
            

            self.model = AutoModelForCausalLM.from_pretrained(
                 model_name,
                 trust_remote_code=True,
                 torch_dtype=torch.float32,
                 device_map={"": "cpu"},
                 cache_dir=cache_dir
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                 model_name,
                 trust_remote_code=True,
                 torch_dtype=torch.float32,
                 device_map={"": "cpu"},
                 cache_dir=cache_dir,
                 attn_implementation="sdpa"
            )
            self.model.eval()
            print("[FLORENCE] Florence-2 loaded successfully! ✅")

        except Exception as e:
            print(f"[FLORENCE ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            print("[FLORENCE WARNING] Continuing in stub mode")

    def caption(self, frame) -> str:
        if self.model is None or self.processor is None:
            return "[Description unavailable]"

        try:
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prompt = "<CAPTION>"
            inputs = self.processor(prompt, images=rgb_frame, return_tensors="pt")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=3
                )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            return self._parse_output(generated_text)
        except Exception as e:
            print(f"[FLORENCE ERROR] Caption failed: {e}")
            return "[Description unavailable]"

    def _parse_output(self, generated_text: str) -> str:
        try:
            if "<CAPTION>" in generated_text:
                start = generated_text.find("<CAPTION>") + len("<CAPTION>")
                end = generated_text.find("<|end|>", start)
                if end > start:
                    return generated_text[start:end].strip()
            return generated_text.strip()
        except:
            return generated_text

    def unload(self):
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            print("[FLORENCE] Model unloaded")