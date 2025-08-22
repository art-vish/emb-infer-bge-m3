from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from contextlib import asynccontextmanager

from src.services.model_service import load_model, clear_gpu_memory
from src.services.batch_service import batch_processor
from src.api.routes import router
from src.core.config import MODEL_NAME, API_TOKEN, MAX_QUEUE_SIZE, BATCH_SIZE, BATCH_TIMEOUT_MS
from src.core.logging_config import get_logger

logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log API configuration
    logger.info("Starting BGE-M3 API", extra={
        "api_token_set": API_TOKEN != "default_token_change_me",
        "model_name": MODEL_NAME,
        "max_queue_size": MAX_QUEUE_SIZE,
        "batch_size": BATCH_SIZE,
        "batch_timeout_ms": BATCH_TIMEOUT_MS
    })
    
    # Load the model at startup
    logger.info("Loading BGE-M3 model", extra={"model_name": MODEL_NAME})
    app.state.encoder = await load_model()
    logger.info("BGE-M3 model loaded successfully")
    
    # Start the batch processor
    app.state.batch_timeout_task = asyncio.create_task(batch_processor.start_timeout_processor())
    logger.info("Batch processor initialized", extra={
        "batch_size": BATCH_SIZE,
        "timeout_ms": BATCH_TIMEOUT_MS
    })
    
    yield  # API is ready to serve requests

    # Clean up resources on shutdown
    logger.info("Shutting down API - releasing resources")
    if hasattr(app.state, "batch_timeout_task"):
        app.state.batch_timeout_task.cancel()
        try:
            await app.state.batch_timeout_task
        except asyncio.CancelledError:
            logger.debug("Batch timeout task cancelled successfully")
    
    if hasattr(app.state, "encoder") and app.state.encoder is not None:
        del app.state.encoder
        logger.debug("Model removed from memory")
    clear_gpu_memory()
    logger.info("API shutdown complete - resources released")
    
app = FastAPI(
    lifespan=lifespan,
    title="emb-infer-bge-m3",
    description="BGE-M3 embedding inference API with selective vector generation. Single endpoint for dense, sparse, and colbert vectors.",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router) 