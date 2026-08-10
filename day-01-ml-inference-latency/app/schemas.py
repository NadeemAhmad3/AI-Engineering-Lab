from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class InferenceInput(BaseModel):
    features: List[float] = Field(
        ..., 
        json_schema_extra={"example": [5.1, 3.5, 1.4, 0.2]},
        description="List of numerical feature values for model input."
    )

class TimingBreakdown(BaseModel):
    request_parsing_ms: float = Field(..., description="Time taken to parse and validate JSON request")
    model_loading_ms: float = Field(..., description="Time taken to load model from disk (0ms if cached in memory)")
    preprocessing_ms: float = Field(..., description="Time taken to preprocess features into array format")
    model_inference_ms: float = Field(..., description="Time taken to execute model.predict()")
    serialization_ms: float = Field(..., description="Time taken to format prediction into JSON response")
    total_pipeline_ms: float = Field(..., description="Total internal server processing time")

class InferenceResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class label")
    probabilities: Optional[List[float]] = Field(None, description="Class probabilities if supported")
    mode: str = Field(..., description="'naive' (reloaded per request) or 'optimized' (loaded on startup)")
    timing: TimingBreakdown
