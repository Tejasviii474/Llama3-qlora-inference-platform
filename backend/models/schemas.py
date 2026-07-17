from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    instruction: str = Field(..., example="What is the difference between a stock and a bond?")
    context: Optional[str] = Field(None, example="Context information here...")
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    use_lora: bool = Field(True, description="Whether to use the fine-tuned LoRA model or the base model")

class ChatResponse(BaseModel):
    response: str
    model_used: str
    latency_ms: float

class EvaluationRequest(BaseModel):
    predictions: List[str]
    references: List[str]

class EvaluationResponse(BaseModel):
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    bertscore_f1: Optional[float] = None
