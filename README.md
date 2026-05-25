# 🏠 House Price Prediction — MLOps Pipeline

[![CI](https://github.com/YOUR_USERNAME/house-price-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/house-price-mlops/actions/workflows/ci.yml)
[![Retrain](https://github.com/YOUR_USERNAME/house-price-mlops/actions/workflows/retrain.yml/badge.svg)](https://github.com/YOUR_USERNAME/house-price-mlops/actions/workflows/retrain.yml)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](./Dockerfile)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange)](http://localhost:5000)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)

End-to-end MLOps pipeline for predicting residential house prices using the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). Built with **FastAPI · MLflow · Docker · GitHub Actions**.

---

## 📐 Architecture

```
Data Ingestion → Preprocessing → Feature Engineering → Model Training → Deployment → Monitoring
    (pandas)      (sklearn)         (pandas)            (XGBoost)      (FastAPI)    (MLflow)
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/house-price-mlops.git
cd house-price-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python -m src.ingest.ingest          # Download & validate data
python -m src.preprocess.preprocess  # Clean & transform
python -m src.features.engineer      # Feature engineering
python -m src.train.train            # Train + log to MLflow
```

### 3. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### 4. Docker Compose (recommended)

```bash
docker compose up --build
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

---

## 📡 API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 7,
    "GrLivArea": 1500,
    "GarageCars": 2,
    "TotalBsmtSF": 900,
    "YearBuilt": 2003,
    "Neighborhood": "NridgHt"
  }'
```

**Response:**
```json
{
  "predicted_price_usd": 214500.0,
  "model_version": "3",
  "run_id": "abc123def456"
}
```

---

## 🗂 Project Structure

```
house-price-mlops/
├── src/
│   ├── ingest/         # Data download & validation
│   ├── preprocess/     # Cleaning & sklearn pipelines
│   ├── features/       # Feature engineering
│   ├── train/          # XGBoost training + MLflow logging
│   ├── api/            # FastAPI app
│   └── monitor/        # Drift detection & alerting
├── tests/              # pytest unit + integration tests
├── .github/workflows/  # CI, retrain, deploy, monitor
├── data/
│   ├── raw/            # Downloaded CSVs (git-ignored)
│   └── processed/      # Feature-engineered data (git-ignored)
├── models/             # Saved model artifacts (git-ignored)
├── notebooks/          # Exploratory analysis
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 📊 Data Sources

| Dataset | URL | Description |
|---------|-----|-------------|
| Ames Housing (train) | [Kaggle via raw GitHub](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv) | 1,460 rows · 81 features |
| Ames Housing (full) | [OpenML #42165](https://www.openml.org/d/42165) | 2,930 rows · 80 features |
| New batch (simulated) | `data/raw/batch_*.csv` | Dropped nightly for retraining |

---

## 🔬 MLflow Tracking

Open the MLflow UI after running any pipeline stage:

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

Every run logs: hyperparameters, RMSE, MAE, R², MAPE, fitted preprocessor, feature schema, and the serialised model.

---

## ⚙️ GitHub Actions Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Push / PR | Lint (ruff), type-check (mypy), pytest |
| `retrain.yml` | Nightly 02:00 UTC | Full pipeline → auto-promote if RMSE improves |
| `deploy.yml` | Tag `v*.*.*` | Build image → push GHCR → SSH redeploy |
| `monitor.yml` | Nightly 06:00 UTC | PSI drift check → open GitHub Issue if alert |

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📦 Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MODEL_NAME` | `house-price-predictor` | Registry model name |
| `MODEL_STAGE` | `Production` | Registry stage to serve |
| `DATA_URL_TRAIN` | *(see config.py)* | Raw training CSV URL |
| `DATA_URL_FULL` | *(see config.py)* | Full dataset URL |
| `DRIFT_PSI_THRESHOLD` | `0.2` | Alert threshold |

---

## 📄 License

MIT
