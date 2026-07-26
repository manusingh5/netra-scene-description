# models/llm_model.py
import os
import torch
import sys

# Reuse cache from Florence-2 setup
cache_dir = os.path.join(os.path.expanduser('~'), 'hf_cache')
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

class LLMModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
        # Check if Phi-3 should be loaded or fallback
        USE_REAL_LLM = True
        
        if USE_REAL_LLM:
            print("[LLM] Loading Phi-3 Mini...")
            
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                model_name = "microsoft/Phi-3-mini-4k-instruct"
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
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
                print("[LLM] Phi-3 Mini loaded successfully! ✅")
                
            except Exception as e:
                print(f"[LLM ERROR] Failed to load model: {e}")
                print("[LLM WARNING] Falling back to smart template mode")
                self.model = None
                self.tokenizer = None
        
        else:
            print("[LLM] Using smart fallback mode (CPU-friendly)")
    
    def generate(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            return self._smart_fallback(prompt)
        
        try:
            chat_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            
            inputs = self.tokenizer(
                chat_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3500
            ).to("cpu")
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            return response if response else self._smart_fallback(prompt)
        
        except Exception as e:
            print(f"[LLM ERROR] Generation failed: {e}")
            return self._smart_fallback(prompt)
    
    def _smart_fallback(self, prompt: str) -> str:
        captions = []
        objects = set()
        texts = set()
        
        for line in prompt.split('\n'):
            line = line.strip()
            if line.startswith('Caption:'):
                cap = line.replace('Caption:', '').strip()
                if cap and cap not in ["[Description unavailable]", "[Skipped for speed]", "None"]:
                    captions.append(cap)
            elif line.startswith('Objects:'):
                objs = line.replace('Objects:', '').strip()
                if objs and objs != "None":
                    for obj in objs.split(','):
                        obj = obj.strip()
                        if obj:
                            objects.add(obj)
        
        narration_parts = []
        if captions:
            narration_parts.append(captions[0])
        
        if objects:
            obj_list = sorted(objects)
            if len(obj_list) == 1:
                narration_parts.append(f"A {obj_list[0]} is visible")
            elif len(obj_list) == 2:
                narration_parts.append(f"A {obj_list[0]} and a {obj_list[1]} are visible")
            else:
                narration_parts.append(f"I can see {', '.join(obj_list[:-1])}, and {obj_list[-1]}")
        
        if narration_parts:
            return ". ".join(narration_parts) + "."
        else:
            return "The video shows a scene with various elements."
    
    def unload(self):
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            import gc
            gc.collect()
            print("[LLM] Model unloaded")