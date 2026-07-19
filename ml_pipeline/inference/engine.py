import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PeftModel
from typing import Optional, List, Dict, Generator
import os

class InferenceEngine:
    def __init__(self, base_model_path: str, lora_weights_path: Optional[str] = None, use_4bit: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        hf_token = os.environ.get("HF_TOKEN")
        print(f"Loading base model from {base_model_path} onto {self.device}")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load Model
        kwargs = {"device_map": "auto", "token": hf_token}
        if use_4bit:
            kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
        
        # Load LoRA weights if provided
        if lora_weights_path and os.path.exists(lora_weights_path):
            print(f"Loading LoRA weights from {lora_weights_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
            
        self.model.eval()

    def _build_prompt(self, instruction: str, context: Optional[str] = None) -> str:
        system_prompt = "You are a highly knowledgeable financial AI assistant. Your goal is to provide accurate, clear, and helpful answers to financial and economic questions."
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        
        user_content = instruction
        if context:
            user_content += f"\n\nContext:\n{context}"
            
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def generate(self, instruction: str, context: Optional[str] = None, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = self._build_prompt(instruction, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode and extract only the assistant's response
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Split by the assistant header and get the content
        try:
            response = decoded_output.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0].strip()
        except IndexError:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return response

if __name__ == "__main__":
    # Test the inference engine with the base model
    engine = InferenceEngine(base_model_path="meta-llama/Meta-Llama-3-8B-Instruct", use_4bit=True)
    res = engine.generate("What is the difference between a stock and a bond?")
    print("Response:", res)
