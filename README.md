# Forest Cover Type Prediction API

A machine learning pipeline for predicting forest cover types using the UCI Covertype dataset. Built with Python, FastAPI, and deployable via Docker/Kubernetes.

## What it does

This project trains multiple classification models on forest coverage data (581,012 samples, 54 features) and serves predictions through a REST API. You can choose between 4 different models at prediction time.

**Models included:**
| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Random Forest | 95.5% | ~3 min |
| Gradient Boosting | 85.1% | ~17 min |
| Decision Tree | 80.2% | ~11 sec |
| Logistic Regression | 71.4% | ~3 min |

## Getting started

### Option 1: Run with Docker (recommended)

```bash
docker pull lazerai/ml-pipeline:latest
docker run -p 8000:8000 lazerai/ml-pipeline:latest
```

Then open http://localhost:8000/docs

### Option 2: Run locally

```bash
# Clone and setup
git clone https://github.com/LAZERAI/ml-pipeline.git
cd ml-pipeline
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Train models (takes ~25 min for all 4)
python scripts/run_pipeline.py

# Start API
uvicorn src.api.main:app --reload
```

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check if API is running |
| `/model/info` | GET | Get info about loaded models |
| `/predict` | POST | Single prediction (pick any model) |
| `/predict/batch` | POST | Batch predictions (up to 100) |

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [2596,51,3,258,0,510,221,232,148,6279], "model": "random_forest"}'
```

## Project structure

```
ml-pipeline/
├── src/
│   ├── api/            # FastAPI app and routes
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Training and evaluation
│   └── utils/          # Config and logging
├── data/               # Raw and processed datasets
├── artifacts/          # Trained model files (.pkl)
├── k8s/                # Kubernetes deployment files
├── configs/            # YAML configuration
└── tests/              # Unit tests
```

## Deployment

### Docker

```bash
docker build -t ml-pipeline .
docker run -p 8000:8000 ml-pipeline
```

### Kubernetes

```bash
kubectl apply -f k8s/
```

This creates:
- 2 replica pods
- LoadBalancer service on port 80
- Horizontal Pod Autoscaler (2-10 pods based on CPU)

## Running tests

```bash
pytest tests/ -v
```

## Tech stack

- Python 3.11
- scikit-learn (ML models)
- FastAPI (REST API)
- Docker & Kubernetes (deployment)
- pandas, numpy (data processing)

## Dataset

Uses the [Covertype dataset](https://archive.ics.uci.edu/ml/datasets/covertype) from UCI ML Repository. Predicts forest cover type (7 classes) from cartographic variables like elevation, slope, and soil type.

## License

MIT
