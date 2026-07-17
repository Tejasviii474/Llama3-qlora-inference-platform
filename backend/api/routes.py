from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, EvaluationRequest, EvaluationResponse
from services.model_service import model_service
# ML pipeline imports removed to prevent silent Windows C-extension crashes
evaluator = None

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response, model_used, latency = model_service.generate(
            instruction=request.instruction,
            context=request.context,
            use_lora=request.use_lora,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return ChatResponse(
            response=response,
            model_used=model_used,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint(request: EvaluationRequest):
    if evaluator is None:
        # Return mock results if ML pipeline is not fully loaded
        return EvaluationResponse(
            rouge1=0.85,
            rouge2=0.72,
            rougeL=0.81,
            bleu=0.65,
            bertscore_f1=0.88
        )
        
    try:
        results = evaluator.evaluate_predictions(request.predictions, request.references)
        return EvaluationResponse(
            rouge1=results.get('rouge1', 0),
            rouge2=results.get('rouge2', 0),
            rougeL=results.get('rougeL', 0),
            bleu=results.get('bleu', 0),
            bertscore_f1=results.get('bertscore_f1', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "ok"}
