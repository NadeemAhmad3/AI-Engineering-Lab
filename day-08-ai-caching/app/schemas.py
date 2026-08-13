from pydantic import BaseModel, Field
from typing import Optional, Any

class QueryRequest(BaseModel):
    text: str = Field(..., description="Input text query or prompt.", json_schema_extra={"example": "What is machine learning?"})
    model_version: Optional[str] = Field("v1.2.0", description="Model version tag for cache invalidation.")

class CacheResponse(BaseModel):
    result: Any = Field(..., description="Model response payload")
    latency_ms: float = Field(..., description="Round-trip latency in ms")
    cache_hit: bool = Field(..., description="Whether response came from cache")
    cache_strategy: str = Field(..., description="Caching strategy ('no-cache', 'exact-cache', 'semantic-cache')")
    similarity_score: Optional[float] = Field(None, description="Cosine similarity score for semantic cache")
