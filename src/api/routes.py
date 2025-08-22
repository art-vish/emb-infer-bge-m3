from fastapi import APIRouter, Depends, HTTPException
import time

from src.models.schemas import EmbeddingRequest, BGEEmbeddingResponse, ModelList, ModelData
from src.services.model_service import process_bge_embeddings
from src.services.batch_service import enqueue_batch_request, get_batch_stats
from src.api.auth import verify_token
from src.core.config import MODEL_NAME, BATCH_SIZE, BATCH_TIMEOUT_MS, MAX_QUEUE_SIZE
from src.core.text_validator import validate_embedding_input, get_input_stats
from src.core.logging_config import get_logger

logger = get_logger("api_routes")

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
    # Validate input texts
    start_time = time.time()
    validated_texts = validate_embedding_input(request.input)
    input_stats = get_input_stats(validated_texts)
    validation_time = time.time() - start_time
    
    logger.info("Input validation completed", extra={
        "validation_time": round(validation_time * 1000, 2),  # ms
        "input_stats": input_stats,
        "return_dense": getattr(request, 'return_dense', True),
        "return_sparse": getattr(request, 'return_sparse', True),
        "return_colbert": getattr(request, 'return_colbert', True)
    })
    
    # Update request with validated input
    request.input = validated_texts
    
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
            "health": "/health",                # Simple health check
            "health_detailed": "/health/detailed", # Detailed health with auth
            "stats": "/stats"                   # Batch statistics
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
    """Simple health check endpoint for load balancers and orchestrators."""
    return {"status": "ok"}

@router.get("/health/detailed")
async def detailed_health_check(token: str = Depends(verify_token)):
    """Detailed health check with model and batch statistics."""
    from src.main import app
    
    # Check model status
    model_status = "loaded" if (hasattr(app.state, "encoder") and app.state.encoder is not None) else "not_loaded"
    
    # Get batch statistics
    batch_stats = get_batch_stats()
    
    health_info = {
        "status": "ok" if model_status == "loaded" else "degraded",
        "timestamp": time.time(),
        "model": {
            "name": MODEL_NAME,
            "status": model_status
        },
        "batching": {
            "enabled": True,
            "batch_size": BATCH_SIZE,
            "timeout_ms": BATCH_TIMEOUT_MS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "stats": batch_stats
        }
    }
    
    logger.info("Detailed health check requested", extra={
        "model_status": model_status,
        "batch_stats": batch_stats
    })
    
    return health_info

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