# Placement Prediction MLOps Project

This is an end-to-end MLOps mini project built using:

- Python
- Scikit-learn
- FastAPI
- MLflow
- Docker
- GitHub

## Features
- Train ML model
- Track experiments with MLflow
- API testing using FastAPI
- Docker containerization
- GitHub version control

## Run Project

### Install Requirements
pip install -r requirements.txt

### Train Model
python train.py

### Run FastAPI
uvicorn app.main:app --reload

### Run MLflow
mlflow ui

### Docker Build
docker build -t placement-mlops .

### Docker Run
docker run -p 8000:8000 placement-mlops



## Run Locally

```bash
git clone https://github.com/buddhisth04/placement-prediction-mlops.git

cd placement-prediction-mlops

docker build -t placement-mlops .

docker run -p 8000:8000 placement-mlops
Open:
http://localhost:8000/docs
