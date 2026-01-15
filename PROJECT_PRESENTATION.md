# Forest Cover Type Prediction - Complete Project Guide

## Project Summary

| Item | Details |
|------|---------|
| **What** | ML API that predicts forest tree types from land measurements |
| **Dataset** | 581,012 samples, 54 features, 7 forest classes |
| **Best Model** | Random Forest - 95.51% accuracy |
| **API** | FastAPI with Swagger UI documentation |
| **Deployment** | Docker container on Kubernetes |
| **GitHub** | https://github.com/LAZERAI/ml-pipeline |
| **Docker Hub** | https://hub.docker.com/r/lazerai/ml-pipeline |

---

## Part 1: What Does Each Technology Do?

### Docker - The Shipping Container

**Problem it solves:** "It works on my machine but not yours"

```
Without Docker:
Your PC: Python 3.11, pandas 2.0    →  Works ✓
Friend's PC: Python 3.9, pandas 1.5  →  Crashes ✗

With Docker:
Your PC:     [Container with Python 3.10, all libraries]  →  Works ✓
Friend's PC: [Same exact container]                       →  Works ✓
Cloud:       [Same exact container]                       →  Works ✓
```

**In this project:** Docker packages our Python code, trained models, and all dependencies into one portable image that runs identically everywhere.

### Kubernetes - The Container Manager

**Problem it solves:** What if 1000 users hit your API at once? What if your container crashes?

```
Without Kubernetes:
- 1 container running
- Container crashes → API down, users angry
- 1000 users → container overloaded, slow responses

With Kubernetes:
- Runs multiple copies (pods) of your container
- One crashes → auto-restarts immediately
- High traffic → auto-creates more pods
- Low traffic → scales down to save resources
```

**In this project:** Kubernetes runs 2 copies of our API, auto-restarts if one dies, and can scale up to 10 pods if traffic increases.

### FastAPI - The Web Framework

**What it does:** Turns your Python function into a web endpoint anyone can call.

```python
# Your ML code:
model.predict(features)  # Only you can run this

# With FastAPI:
POST /predict  # Anyone on the internet can call this
```

**In this project:** FastAPI serves our trained model as a REST API with automatic Swagger documentation at `/docs`.

---

## Part 2: The 4 Trained Models

We trained 4 different algorithms on the same data:

| Model | Accuracy | Training Time | When to Use |
|-------|----------|---------------|-------------|
| **Random Forest** | 95.51% | 3 min | Best accuracy, use as default |
| **Gradient Boosting** | 85.1% | 17 min | Good but slower to train |
| **Decision Tree** | 80.2% | 11 sec | Fast training, lower accuracy |
| **Logistic Regression** | 71.4% | 3 min | Simple baseline |

The API lets you pick which model to use at prediction time.

---

## Part 3: All Commands You Need to Know

### Running Locally (Python)

```powershell
# 1. Clone the project
git clone https://github.com/LAZERAI/ml-pipeline.git
cd ml-pipeline

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train all 4 models (takes ~25 minutes)
python scripts/run_pipeline.py

# 5. Start the API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 6. Open browser: http://localhost:8000/docs
```

### Running with Docker

```powershell
# Option A: Pull from Docker Hub (pre-built, fastest)
docker pull lazerai/ml-pipeline:latest
docker run -p 8000:8000 lazerai/ml-pipeline:latest

# Option B: Build yourself
docker build -t ml-pipeline .
docker run -p 8000:8000 ml-pipeline

# Open browser: http://localhost:8000/docs
```

### Docker Commands Explained

```powershell
# See all images on your computer
docker images

# See running containers
docker ps

# See all containers (including stopped)
docker ps -a

# Stop a container
docker stop <container_id>

# Remove a container
docker rm <container_id>

# Remove an image
docker rmi <image_name>

# View container logs
docker logs <container_id>

# Enter inside a running container
docker exec -it <container_id> /bin/bash
```

### Running with Kubernetes

```powershell
# 1. Make sure Kubernetes is enabled in Docker Desktop
#    Settings → Kubernetes → Enable Kubernetes → Apply

# 2. Apply all Kubernetes configs
kubectl apply -f k8s/

# 3. Check if pods are running
kubectl get pods

# 4. Check the service
kubectl get services

# 5. Open browser: http://localhost/docs (port 80)
```

### Kubernetes Commands Explained

```powershell
# See all pods (containers)
kubectl get pods

# See detailed pod info
kubectl describe pod <pod_name>

# See all services
kubectl get services

# See all deployments
kubectl get deployments

# See pod logs
kubectl logs <pod_name>

# Enter a pod
kubectl exec -it <pod_name> -- /bin/bash

# Scale manually to 3 replicas
kubectl scale deployment ml-pipeline --replicas=3

# Restart deployment (pulls new image)
kubectl rollout restart deployment ml-pipeline

# Delete everything
kubectl delete -f k8s/

# See Horizontal Pod Autoscaler status
kubectl get hpa
```

### Pushing to Docker Hub

```powershell
# 1. Login to Docker Hub
docker login

# 2. Tag your image with your username
docker tag ml-pipeline:latest YOUR_USERNAME/ml-pipeline:latest
docker tag ml-pipeline:latest YOUR_USERNAME/ml-pipeline:v1.0.0

# 3. Push to Docker Hub
docker push YOUR_USERNAME/ml-pipeline:latest
docker push YOUR_USERNAME/ml-pipeline:v1.0.0

# Others can now pull your image:
docker pull YOUR_USERNAME/ml-pipeline:latest
```

### Pushing to GitHub

```powershell
# 1. Initialize git (only once)
git init
git remote add origin https://github.com/YOUR_USERNAME/ml-pipeline.git

# 2. Stage all files
git add -A

# 3. Commit with a message
git commit -m "Your commit message here"

# 4. Push to GitHub
git push origin main

# Common git commands
git status          # See what changed
git diff            # See exact changes
git log --oneline   # See commit history
git pull origin main  # Get latest from GitHub
```

---

## Part 4: API Endpoints

Open http://localhost:8000/docs (or http://localhost/docs with K8s) to see Swagger UI.

### GET /health
Check if API is running.
```json
Response: {"status": "healthy", "model_loaded": true}
```

### GET /model/info
Get info about loaded models.
```json
Response: {
  "models_loaded": ["random_forest", "logistic_regression", "decision_tree", "gradient_boosting"],
  "default_model": "random_forest",
  "feature_count": 54
}
```

### POST /predict
Make a single prediction. You can choose which model to use.

```json
Request: {
  "features": [2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, ...],
  "model": "random_forest"
}

Response: {
  "prediction": 2,
  "prediction_label": "Lodgepole Pine",
  "probability": 0.89,
  "model_used": "random_forest"
}
```

### POST /predict/batch
Predict multiple samples at once (up to 100).

```json
Request: {
  "samples": [
    {"features": [2596, 51, 3, ...]},
    {"features": [2590, 56, 2, ...]}
  ]
}
```

---

## Part 5: Project Files Explained

```
ml-pipeline/
├── src/
│   ├── api/
│   │   ├── main.py        # FastAPI app setup, loads models on startup
│   │   ├── routes.py      # All 4 endpoints defined here
│   │   └── schemas.py     # Request/response data structures
│   ├── data/
│   │   ├── ingestion.py   # Downloads/loads the dataset
│   │   ├── validation.py  # Checks data quality
│   │   └── preprocessing.py # Splits into train/test
│   ├── models/
│   │   ├── train.py       # Training logic for all 4 models
│   │   └── evaluate.py    # Calculates accuracy, confusion matrix
│   └── utils/
│       ├── config.py      # Reads YAML configuration
│       └── logger.py      # Logging setup
│
├── data/
│   ├── raw/               # Original dataset (covtype_data.csv)
│   └── processed/         # Train/test splits (X_train, y_train, etc.)
│
├── artifacts/             # Trained model files (.pkl)
│   ├── model_random_forest.pkl
│   ├── model_gradient_boosting.pkl
│   ├── model_decision_tree.pkl
│   └── model_logistic_regression.pkl
│
├── k8s/                   # Kubernetes configuration
│   ├── deployment.yaml    # Defines pods and containers
│   ├── service.yaml       # Exposes pods to network
│   ├── hpa.yaml           # Auto-scaling rules
│   └── configmap.yaml     # Environment variables
│
├── configs/
│   ├── model_config.yaml  # Model hyperparameters
│   └── pipeline_config.yaml # Pipeline settings
│
├── Dockerfile             # Instructions to build container
├── docker-compose.yaml    # Run with docker-compose up
├── requirements.txt       # Python dependencies
└── README.md              # Quick start guide
```

---

## Part 6: The Dataset

**Name:** Forest Covertype Dataset (UCI Machine Learning Repository)

**What it predicts:** Type of trees in a forest based on land characteristics.

**Size:** 581,012 samples (rows), 54 features (columns)

**Features include:**
- Elevation (meters above sea level)
- Aspect (compass direction of slope)
- Slope (steepness in degrees)
- Distance to water sources
- Distance to roads
- Hillshade values (sunlight)
- Wilderness area (4 areas in Colorado)
- Soil type (40 types, one-hot encoded)

**7 Forest Types (Classes):**
| Class | Type | Description |
|-------|------|-------------|
| 1 | Spruce/Fir | High elevation evergreens |
| 2 | Lodgepole Pine | Most common (48% of data) |
| 3 | Ponderosa Pine | Lower elevation, drier |
| 4 | Cottonwood/Willow | Near water |
| 5 | Aspen | Deciduous, turns gold in fall |
| 6 | Douglas-fir | Mixed forests |
| 7 | Krummholz | Stunted trees at treeline |

---

## Part 7: How to Present This

### Opening (30 seconds)
> "I built a machine learning system that predicts forest cover types from geographical data. It has 95% accuracy on 580,000 samples and runs as a scalable API."

### Demo Flow (3 minutes)
1. Open http://localhost/docs
2. Click `/health` → Execute → "API is running"
3. Click `/model/info` → Execute → "Shows 4 loaded models"
4. Click `/predict` → Enter sample features → Execute → "Predicted Lodgepole Pine with 89% confidence"
5. Change model to "logistic_regression" → Execute → "Same prediction but different confidence"

### Technical Points to Mention
- "Docker ensures it runs the same on any computer"
- "Kubernetes auto-scales from 2 to 10 pods based on traffic"
- "FastAPI auto-generates this documentation from code"
- "Model is 95% accurate on 116,000 test samples"

### Closing
> "This is how companies like Netflix and Uber deploy their ML models. Containerized, scalable, and production-ready."

---

## Part 8: Quick Reference Card

| What | Command |
|------|---------|
| **Run locally** | `uvicorn src.api.main:app --reload` |
| **Run Docker** | `docker run -p 8000:8000 lazerai/ml-pipeline` |
| **Deploy K8s** | `kubectl apply -f k8s/` |
| **Check pods** | `kubectl get pods` |
| **View logs** | `kubectl logs <pod_name>` |
| **Scale up** | `kubectl scale deployment ml-pipeline --replicas=5` |
| **Open docs** | http://localhost:8000/docs or http://localhost/docs |

---

## Links

- **GitHub:** https://github.com/LAZERAI/ml-pipeline
- **Docker Hub:** https://hub.docker.com/r/lazerai/ml-pipeline
- **Dataset:** https://archive.ics.uci.edu/ml/datasets/covertype
