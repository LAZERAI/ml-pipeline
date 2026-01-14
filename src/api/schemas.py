"""
API Schemas Module
------------------
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum


# Example features for Forest Covtype (54 features)
EXAMPLE_FEATURES = [
    2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279,  # Elevation, Aspect, Slope, etc.
    1, 0, 0, 0,  # Wilderness Area (4 binary)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Soil Type (40 binary)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
]


class ModelType(str, Enum):
    """Supported classifier types."""
    random_forest = "random_forest"
    logistic_regression = "logistic_regression"
    decision_tree = "decision_tree"
    gradient_boosting = "gradient_boosting"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(
        ..., 
        description="Current health status",
        json_schema_extra={"example": "healthy"}
    )
    message: str = Field(
        ..., 
        description="Status message",
        json_schema_extra={"example": "Service is running"}
    )


class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint."""
    model_name: str = Field(
        ..., 
        description="Name of the loaded model algorithm",
        json_schema_extra={"example": "random_forest"}
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether a model is currently loaded",
        json_schema_extra={"example": True}
    )
    training_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Training metadata including samples, duration, and parameters"
    )


class PredictionRequest(BaseModel):
    """
    Single prediction request.
    
    Features array must contain 54 values matching the Covtype schema:
    10 quantitative features followed by 44 binary indicators.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"features": EXAMPLE_FEATURES, "model": "random_forest"}
        }
    )
    features: List[float] = Field(
        ..., 
        description="54-element feature vector",
        min_length=54,
        max_length=54,
        json_schema_extra={"example": EXAMPLE_FEATURES}
    )
    model: Optional[ModelType] = Field(
        default=ModelType.random_forest,
        description="Classifier to use. Defaults to random_forest."
    )


class PredictionResponse(BaseModel):
    """Prediction result."""
    prediction: int = Field(
        ..., 
        description="Predicted class ID (1-7)",
        ge=1,
        le=7,
        json_schema_extra={"example": 2}
    )
    prediction_label: Optional[str] = Field(
        None, 
        description="Class label",
        json_schema_extra={"example": "Lodgepole Pine"}
    )
    probability: Optional[List[float]] = Field(
        None, 
        description="Per-class probabilities (may not be calibrated)",
        json_schema_extra={"example": [0.27, 0.56, 0.0, 0.0, 0.05, 0.12, 0.0]}
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"instances": [EXAMPLE_FEATURES, EXAMPLE_FEATURES]}
        }
    )
    instances: List[List[float]] = Field(
        ..., 
        description="List of 54-element feature vectors",
        json_schema_extra={"example": [EXAMPLE_FEATURES, EXAMPLE_FEATURES]}
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction result."""
    predictions: List[int] = Field(
        ..., 
        description="Predicted class IDs",
        json_schema_extra={"example": [2, 1]}
    )
    count: int = Field(
        ..., 
        description="Number of samples processed",
        json_schema_extra={"example": 2}
    )


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    error: str = Field(
        ..., 
        description="Error type or message",
        json_schema_extra={"example": "Model not loaded"}
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error description"
    )
