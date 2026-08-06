"""FastAPI backend for inference"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Optional

from inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lumina API",
    description="API for Lumina language model",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    strategy: str = "top_p"


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    prompt: str
    generated_text: str
    length: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """
    Initialize pipeline on startup.
    """
    global pipeline
    logger.info("Initializing inference pipeline...")
    pipeline = InferencePipeline(device='auto')
    logger.info("Inference pipeline ready")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline.model is not None
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    Generate text from prompt.
    """
    if pipeline.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if pipeline.tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        # Set generation parameters
        pipeline.temperature = request.temperature
        pipeline.top_k = request.top_k
        pipeline.top_p = request.top_p
        
        # Generate
        generated_text = pipeline.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            strategy=request.strategy
        )
        
        return GenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            length=len(generated_text)
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-model")
async def load_model(checkpoint_path: str):
    """
    Load model from checkpoint.
    """
    try:
        pipeline.load_model(checkpoint_path)
        return {"status": "success", "message": f"Loaded model from {checkpoint_path}"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def model_info():
    """
    Get model information.
    """
    if pipeline.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    config = pipeline.model.get_config()
    return {
        "status": "loaded",
        "config": config,
        "device": str(pipeline.device)
    }
