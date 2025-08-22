from fastapi import APIRouter, Depends, HTTPException
import time

from src.models.schemas import EmbeddingRequest, BGEEmbeddingResponse, ModelList, ModelData
from src.services.model_service import process_bge_embeddings
from src.services.batch_service import enqueue_batch_request, get_batch_stats
from src.api.auth import verify_token
from src.core.config import MODEL_NAME, BATCH_SIZE, BATCH_TIMEOUT_MS, MAX_QUEUE_SIZE

router = APIRouter()

# ============================================
# ОСНОВНОЙ EMBEDDING ENDPOINT
# ============================================

@router.post("/v1/embeddings", response_model=BGEEmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, token: str = Depends(verify_token)):
    """
    BGE-M3 endpoint для получения выбранных типов векторов.
    Поддерживает параметры return_dense, return_sparse, return_colbert.
    
    Примеры использования:
    - Только dense: {"input": "text", "return_dense": true, "return_sparse": false, "return_colbert": false}
    - Только sparse: {"input": "text", "return_dense": false, "return_sparse": true, "return_colbert": false}  
    - Только colbert: {"input": "text", "return_dense": false, "return_sparse": false, "return_colbert": true}
    - Все типы: {"input": "text"} (по умолчанию все включены)
    - Массив текстов: {"input": ["text1", "text2", "text3"]}
    """
    return await enqueue_batch_request(request, process_bge_embeddings)

# ============================================
# СЛУЖЕБНЫЕ ENDPOINTS
# ============================================

@router.get("/v1/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models."""
    models = [ModelData(id=MODEL_NAME)]
    return ModelList(data=models)

@router.get("/v1/models/{model_id}")
async def get_model(model_id: str, token: str = Depends(verify_token)):
    """Get model information."""
    if model_id != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return ModelData(id=MODEL_NAME)

@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "emb-infer-bge-m3",
        "description": "BGE-M3 embedding inference API with selective vector generation",
        "version": "3.0",
        "model": MODEL_NAME,
        "docs_url": "/docs",
        "endpoints": {
            "embeddings": "/v1/embeddings",     # BGE-M3 selective vectors
            "models": "/v1/models",
            "health": "/health",
            "stats": "/stats"
        },
        "supported_vectors": ["dense", "sparse", "colbert"],
        "batching": {
            "enabled": True,
            "batch_size": BATCH_SIZE,
            "timeout_ms": BATCH_TIMEOUT_MS,
            "max_queue_size": MAX_QUEUE_SIZE
        }
    }

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    from src.main import app
    
    if not hasattr(app.state, "encoder") or app.state.encoder is None:
        return {
            "status": "error",
            "message": "Model not loaded",
            "timestamp": time.time()
        }
    
    batch_stats = get_batch_stats()
    
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "batching": {
            "enabled": True,
            "batch_size": BATCH_SIZE,
            "timeout_ms": BATCH_TIMEOUT_MS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "stats": batch_stats
        },
        "timestamp": time.time()
    }

@router.get("/stats")
async def get_stats():
    """Get detailed batching statistics."""
    return {
        "batching": get_batch_stats(),
        "config": {
            "batch_size": BATCH_SIZE,
            "timeout_ms": BATCH_TIMEOUT_MS,
            "max_queue_size": MAX_QUEUE_SIZE
        },
        "timestamp": time.time()
    }