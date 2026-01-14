"""
API Routes Module
-----------------
FastAPI route definitions.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
from typing import List, Dict

from .schemas import (
    HealthResponse, 
    ModelInfoResponse,
    PredictionRequest, 
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelType
)

# Create router
router = APIRouter()

# Global model references - dict of all loaded models
loaded_models: Dict[str, object] = {}
default_model_name = "random_forest"

# Forest Cover Type labels
COVER_TYPE_LABELS = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}


def set_loaded_models(models: Dict[str, object]):
    """Set the loaded models dictionary."""
    global loaded_models
    loaded_models = models


def get_model(model_name: str):
    """Get a specific model by name."""
    if model_name in loaded_models:
        return loaded_models[model_name]
    return None


@router.get(
    "/health", 
    response_model=HealthResponse, 
    tags=["Health"],
    summary="Health check",
    description="Returns API health status. Used for liveness and readiness probes.",
    responses={
        200: {
            "description": "Service is available",
            "content": {
                "application/json": {
                    "example": {"status": "healthy", "message": "Service is running"}
                }
            }
        }
    }
)
async def health_check():
    """Return health status."""
    return HealthResponse(
        status="healthy",
        message="Service is running"
    )


@router.get(
    "/model/info", 
    response_model=None, 
    tags=["Model"],
    summary="List loaded models",
    description="Returns metadata for all models currently loaded in memory, including training parameters.",
    responses={
        200: {
            "description": "Model metadata",
            "content": {
                "application/json": {
                    "example": {
                        "available_models": ["random_forest", "logistic_regression", "decision_tree", "gradient_boosting"],
                        "default_model": "random_forest",
                        "models_count": 4,
                        "models_info": {
                            "random_forest": {"loaded": True},
                            "logistic_regression": {"loaded": True}
                        }
                    }
                }
            }
        }
    }
)
async def get_model_info():
    """Return metadata for loaded models."""
    if not loaded_models:
        return {
            "available_models": [],
            "default_model": None,
            "models_count": 0,
            "models_info": {}
        }
    
    models_info = {}
    for name, trainer in loaded_models.items():
        models_info[name] = {
            "loaded": trainer.model is not None,
            "training_info": trainer.training_info if hasattr(trainer, 'training_info') else None
        }
    
    return {
        "available_models": list(loaded_models.keys()),
        "default_model": default_model_name,
        "models_count": len(loaded_models),
        "models_info": models_info
    }


@router.post(
    "/predict", 
    response_model=PredictionResponse, 
    tags=["Predictions"],
    summary="Single prediction",
    description="""
Predict forest cover type for a single sample.

### Model Selection

Specify `model` in the request body to select a classifier. Defaults to `random_forest`.

### Input Format

The `features` array must contain exactly 54 values in this order:

| Index | Feature |
|-------|---------|
| 0 | Elevation (meters) |
| 1 | Aspect (degrees azimuth) |
| 2 | Slope (degrees) |
| 3-5 | Distance to hydrology, roadways |
| 6-8 | Hillshade at 9am, noon, 3pm (0-255) |
| 9 | Distance to fire points |
| 10-13 | Wilderness area (one-hot, 4 cols) |
| 14-53 | Soil type (one-hot, 40 cols) |

### Output

Returns predicted class (1-7), label, and class probabilities if supported by the model.
    """,
    responses={
        200: {
            "description": "Prediction result",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 2,
                        "prediction_label": "Lodgepole Pine",
                        "probability": [0.27, 0.56, 0.0, 0.0, 0.05, 0.12, 0.0]
                    }
                }
            }
        },
        503: {
            "description": "Requested model not loaded",
            "content": {
                "application/json": {
                    "example": {"detail": "Model 'random_forest' not loaded"}
                }
            }
        }
    }
)
async def predict(request: PredictionRequest):
    """Run inference on a single feature vector."""
    # Get the requested model (default to random_forest)
    model_name = request.model.value if request.model else default_model_name
    
    # Get the model trainer
    model_trainer = get_model(model_name)
    
    if model_trainer is None or model_trainer.model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model '{model_name}' not loaded. Available: {list(loaded_models.keys())}"
        )
    
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model_trainer.predict(features)[0]
        
        # Get label (classes are 1-7, but array is 0-indexed)
        prediction_label = COVER_TYPE_LABELS.get(int(prediction), None)
        
        # Get probabilities if available
        probability = None
        if hasattr(model_trainer.model, 'predict_proba'):
            try:
                proba = model_trainer.predict_proba(features)[0]
                probability = proba.tolist()
            except:
                pass
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            probability=probability
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/predict/batch", 
    response_model=BatchPredictionResponse, 
    tags=["Predictions"],
    summary="Batch prediction",
    description="""
Predict forest cover types for multiple samples in a single request.

Batch inference avoids per-request overhead and is recommended when classifying more than a few samples.
Each element in `instances` must be a 54-element feature array.

Note: Uses the default model (random_forest). Model selection is not supported for batch requests.
    """,
    responses={
        200: {
            "description": "Batch prediction result",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [2, 1, 5, 2],
                        "count": 4
                    }
                }
            }
        }
    }
)
async def predict_batch(request: BatchPredictionRequest):
    """Run inference on multiple feature vectors."""
    if model_trainer is None or model_trainer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.instances)
        
        # Make predictions
        predictions = model_trainer.predict(features)
        
        return BatchPredictionResponse(
            predictions=[int(p) for p in predictions],
            count=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
