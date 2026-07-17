import time
# ML pipeline imports removed to prevent silent Windows C-extension crashes
HAS_ML_PIPELINE = False

class ModelService:
    def __init__(self):
        self.base_engine = None
        self.lora_engine = None
        self.is_initialized = False

    def initialize(self):
        if not HAS_ML_PIPELINE:
            print("Running without ML pipeline loaded (mock mode)")
            self.is_initialized = True
            return

        base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        lora_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "llama3-finance")
        
        try:
            # Note: In a real environment with limited VRAM, you wouldn't load both simultaneously 
            # or you'd use a dynamic LoRA adapter switching mechanism (like vLLM supports).
            # For simplicity in this demo, we assume lazy loading or a mock if GPU fails.
            self.base_engine = InferenceEngine(base_model_path=base_model_path, use_4bit=True)
            if os.path.exists(lora_weights_path):
                self.lora_engine = InferenceEngine(base_model_path=base_model_path, lora_weights_path=lora_weights_path, use_4bit=True)
            self.is_initialized = True
        except Exception as e:
            print(f"Failed to initialize models: {e}")
            self.is_initialized = True # Fallback to mock

    def generate(self, instruction: str, context: str = None, use_lora: bool = True, **kwargs):
        start_time = time.time()
        
        if not HAS_ML_PIPELINE or (use_lora and self.lora_engine is None) or (not use_lora and self.base_engine is None):
            # Mock response for testing UI
            time.sleep(1.5)
            response = f"[Mock] Response to: {instruction}\nContext provided: {'Yes' if context else 'No'}\nModel used: {'LoRA Fine-tuned' if use_lora else 'Base Model'}"
            model_used = "mock-model"
        else:
            engine = self.lora_engine if use_lora else self.base_engine
            response = engine.generate(instruction=instruction, context=context, **kwargs)
            model_used = "llama3-lora" if use_lora else "llama3-base"
            
        latency = (time.time() - start_time) * 1000
        return response, model_used, latency

model_service = ModelService()
