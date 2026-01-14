"""
FastAPI Main Application
------------------------
Main entry point for the ML Pipeline API.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_loaded_models
from src.models.train import ModelTrainer
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

# Load configuration
config = Config()

# Create FastAPI app
app = FastAPI(
    title="Forest Cover Type Prediction API",
    description="""
Classification API for predicting forest cover types from cartographic variables.

## Overview

This service exposes four trained classifiers for the UCI Forest Covtype dataset.
Clients can select which model to use at prediction time.

### Available Models

| Model | Notes |
|-------|-------|
| `random_forest` | Default. 100 estimators. Generally highest accuracy. |
| `gradient_boosting` | 100 estimators, depth 3. Slower inference. |
| `decision_tree` | Single tree. Fastest inference, lower accuracy. |
| `logistic_regression` | Linear model. May not converge on all feature distributions. |

### Dataset

- Source: UCI ML Repository (Covtype)
- Samples: 581,012
- Features: 54 (10 quantitative, 44 binary)
- Classes: 7 forest cover types

### Target Classes

| ID | Cover Type |
|----|------------|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

### Limitations

- Models were trained on an 80/20 split without hyperparameter tuning.
- Probability calibration was not performed; reported probabilities may not reflect true likelihoods.
- Input features are expected to match training schema exactly (54 values, same order).
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Liveness and readiness probes.",
        },
        {
            "name": "Model",
            "description": "Metadata for loaded models.",
        },
        {
            "name": "Predictions",
            "description": "Inference endpoints for single and batch predictions.",
        },
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Available model names
MODEL_NAMES = ['random_forest', 'logistic_regression', 'decision_tree', 'gradient_boosting']

# Try to load all models on startup
@app.on_event("startup")
async def load_models():
    """Load all trained models on startup."""
    logger.info("Starting ML Pipeline API...")
    logger.info("Loading all available models...")
    
    loaded_models = {}
    
    for model_name in MODEL_NAMES:
        model_path = os.path.join("artifacts", f"model_{model_name}.pkl")
        
        # Fallback to model.pkl for backwards compatibility
        if model_name == 'random_forest' and not os.path.exists(model_path):
            model_path = os.path.join("artifacts", "model.pkl")
        
        if os.path.exists(model_path):
            try:
                trainer = ModelTrainer()
                trainer.load_model(model_path)
                loaded_models[model_name] = trainer
                logger.info(f"Loaded {model_name} from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    set_loaded_models(loaded_models)
    
    logger.info(f"Loaded {len(loaded_models)} model(s): {list(loaded_models.keys())}")
    logger.info("API startup complete")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ML Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
