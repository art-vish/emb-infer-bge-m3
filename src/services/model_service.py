import time
import os
import gc
import torch
import asyncio
import numpy as np

from src.core.config import LOCAL_MODEL_PATH, MODEL_NAME
from src.models.schemas import EmbeddingRequest, BGEEmbeddingData, BGEEmbeddingResponse
from src.core.logging_config import get_logger

logger = get_logger("model_service")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

async def load_model():
    """Load the BGE-M3 model at startup."""
    from FlagEmbedding import BGEM3FlagModel
    
    logger.info("Starting BGE-M3 model loading", extra={"model_name": MODEL_NAME})
    start_load_time = time.time()
    clear_gpu_memory()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Device selected for model", extra={"device": device})
        
        # Check if local path contains model files
        config_path = os.path.join(LOCAL_MODEL_PATH, "config.json")
        if os.path.exists(config_path):
            logger.info("Loading model from local path", extra={"path": LOCAL_MODEL_PATH})
            model_path = LOCAL_MODEL_PATH
        else:
            logger.info("Local model not found, downloading from Hugging Face", extra={"model": "BAAI/bge-m3"})
            model_path = "BAAI/bge-m3"
        
        # Load the BGE-M3 model
        encoder = BGEM3FlagModel(
            model_path,
            use_fp16=True,
            device=device
        )
        
        load_time = time.time() - start_load_time
        logger.info("BGE-M3 model loaded successfully", extra={
            "load_time": round(load_time, 2),
            "device": device,
            "model_name": MODEL_NAME
        })
        
        return encoder
    except Exception as e:
        logger.error("Failed to load BGE-M3 model", extra={"error": str(e)}, exc_info=True)
        raise


async def process_bge_embeddings(request: EmbeddingRequest, encoder):
    """
    Process BGE-M3 embeddings with selective vector types.
    Supports return_dense, return_sparse, return_colbert parameters.
    """
    try:
        # Handle input
        if isinstance(request.input, str):
            inputs = [request.input]
        elif isinstance(request.input, list):
            if not request.input:
                # Определяем запрошенные типы векторов даже для пустого массива
                requested_types = []
                if getattr(request, 'return_dense', True):
                    requested_types.append("dense")
                if getattr(request, 'return_sparse', True):
                    requested_types.append("sparse")
                if getattr(request, 'return_colbert', True):
                    requested_types.append("colbert")
                
                return BGEEmbeddingResponse(
                    data=[],
                    model=request.model,
                    usage={"prompt_tokens": 0, "total_tokens": 0},
                    embedding_types=requested_types
                )
            inputs = request.input
        else:
            raise ValueError("Invalid input format")
        
        # Get vector type selection parameters
        return_dense = getattr(request, 'return_dense', True)
        return_sparse = getattr(request, 'return_sparse', True) 
        return_colbert = getattr(request, 'return_colbert', True)
        
        # Validate that at least one vector type is requested
        if not any([return_dense, return_sparse, return_colbert]):
            raise ValueError("At least one vector type must be requested (return_dense, return_sparse, or return_colbert)")
        
        # Generate all embeddings (BGE-M3 always generates all types)
        start_time = time.time()
        embeddings_result = encoder.encode(
            inputs,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        
        # Extract all vectors
        dense_vecs = embeddings_result.get('dense_vecs', [])
        sparse_vecs = embeddings_result.get('lexical_weights', [])
        colbert_vecs = embeddings_result.get('colbert_vecs', [])
        
        # Count tokens (approximate)
        total_tokens = sum(len(text.split()) * 1.3 for text in inputs)
        
        # Build list of requested embedding types
        requested_types = []
        if return_dense:
            requested_types.append("dense")
        if return_sparse:
            requested_types.append("sparse")
        if return_colbert:
            requested_types.append("colbert")
        
        # Format response with only requested vector types
        data = []
        for i in range(len(inputs)):
            # Process dense vectors (only if requested)
            dense_embedding = None
            if return_dense and i < len(dense_vecs) and dense_vecs[i] is not None:
                dense_embedding = dense_vecs[i].tolist() if isinstance(dense_vecs[i], np.ndarray) else dense_vecs[i]
            
            # Process sparse vectors (only if requested)  
            sparse_embedding = None
            if return_sparse and i < len(sparse_vecs) and sparse_vecs[i] is not None:
                if isinstance(sparse_vecs[i], dict):
                    sparse_embedding = {int(k): float(v) for k, v in sparse_vecs[i].items()}
                else:
                    sparse_embedding = {}
            
            # Process colbert vectors (only if requested)
            colbert_embedding = None
            if return_colbert and i < len(colbert_vecs) and colbert_vecs[i] is not None:
                if isinstance(colbert_vecs[i], np.ndarray):
                    colbert_embedding = colbert_vecs[i].tolist()
                elif isinstance(colbert_vecs[i], list):
                    colbert_embedding = colbert_vecs[i]
            
            data.append(BGEEmbeddingData(
                index=i,
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                colbert_embedding=colbert_embedding
            ))
        
        compute_time = time.time() - start_time
        vector_types_str = "+".join(requested_types)
        logger.info("BGE-M3 embeddings generated", extra={
            "count": len(data),
            "vector_types": vector_types_str,
            "compute_time": round(compute_time, 2)
        })
        
        return BGEEmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": int(total_tokens),
                "total_tokens": int(total_tokens)
            },
            embedding_types=requested_types
        )
    except Exception as e:
        logger.error("Failed to generate BGE embeddings", extra={"error": str(e)}, exc_info=True)
        raise