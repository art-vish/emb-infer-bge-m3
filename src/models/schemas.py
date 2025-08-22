from pydantic import BaseModel
from typing import List, Union, Dict, Optional, Any
import time

from src.core.config import MODEL_NAME

# BGE-M3 request model
class EmbeddingRequest(BaseModel):
    model: str = MODEL_NAME
    input: Union[str, List[str]] = []
    # BGE-M3 vector type selection
    return_dense: Optional[bool] = True
    return_sparse: Optional[bool] = True
    return_colbert: Optional[bool] = True
    # Additional fields
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None
    
    class Config:
        extra = "allow"

class BGEEmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    dense_embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None
    colbert_embedding: Optional[List[List[float]]] = None

class BGEEmbeddingResponse(BaseModel):
    data: List[BGEEmbeddingData]
    model: str
    object: str = "list"
    usage: Dict[str, int]
    embedding_types: List[str] = ["dense", "sparse", "colbert"]

# Model information
class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "organization"
    permission: List[Dict[str, Any]] = []
    root: str = MODEL_NAME
    parent: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelData] 