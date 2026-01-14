# ML Pipeline

Production-Grade Machine Learning Pipeline with FastAPI, Docker & Kubernetes

## 🚀 Quick Start

### 1. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python scripts/run_pipeline.py
```

### 4. Start the API
```bash
python -m src.api.main
# Or
uvicorn src.api.main:app --reload
```

### 5. Access API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 Project Structure

```
ml-pipeline/
├── src/
│   ├── data/           # Data pipeline modules
│   ├── models/         # ML model training & evaluation
│   ├── api/            # FastAPI application
│   └── utils/          # Utilities (config, logging)
├── data/               # Data storage
├── artifacts/          # Trained models
├── logs/               # Application logs
├── tests/              # Unit tests
├── k8s/                # Kubernetes manifests
├── configs/            # Configuration files
├── Dockerfile
└── requirements.txt
```

## 🧪 Running Tests
```bash
pytest tests/ -v
```

## 🐳 Docker
```bash
docker build -t ml-pipeline .
docker run -p 8000:8000 ml-pipeline
```

## ☸️ Kubernetes
```bash
kubectl apply -f k8s/
```

## 📝 License
MIT
