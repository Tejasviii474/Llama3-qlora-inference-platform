from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from services.model_service import model_service
import uvicorn
import os

app = FastAPI(
    title="Llama 3 Fine-Tuning API",
    description="API for evaluating and inferencing on fine-tuned Llama 3 models.",
    version="1.0.0"
)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    print("Initializing Model Service...")
    # This will load the weights or setup mock mode
    model_service.initialize()
    print("Backend API started successfully.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
