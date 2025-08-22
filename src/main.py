from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from contextlib import asynccontextmanager

from src.services.model_service import load_model, clear_gpu_memory
from src.services.batch_service import batch_processor
from src.api.routes import router
from src.core.config import MODEL_NAME, API_TOKEN, MAX_QUEUE_SIZE, BATCH_SIZE, BATCH_TIMEOUT_MS

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Печать информации о конфигурации
    print("\n--- API Configuration ---")
    print(f"API_TOKEN is {'properly set' if API_TOKEN != 'default_token_change_me' else 'using default value - CHANGE THIS!'}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"MAX_QUEUE_SIZE: {MAX_QUEUE_SIZE}")
    print(f"BATCHING: enabled (size={BATCH_SIZE}, timeout={BATCH_TIMEOUT_MS}ms)")
    print("-------------------------\n")
    
    # Load the model at startup
    app.state.encoder = await load_model()
    
    # Start the batch processor
    app.state.batch_timeout_task = asyncio.create_task(batch_processor.start_timeout_processor())
    print(f"Batch processor initialized")
    
    yield  # API is ready to serve requests

    # Clean up resources on shutdown
    print("Releasing resources...")
    if hasattr(app.state, "batch_timeout_task"):
        app.state.batch_timeout_task.cancel()
        try:
            await app.state.batch_timeout_task
        except asyncio.CancelledError:
            pass
    
    if hasattr(app.state, "encoder") and app.state.encoder is not None:
        del app.state.encoder
    clear_gpu_memory()
    print("Resources released.")
    
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