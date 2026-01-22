# MLOps Churn Prediction Project

A production-ready MLOps pipeline for customer churn prediction with automated training, deployment, monitoring, and model versioning capabilities.

## 📋 Overview

This project implements an end-to-end machine learning operations (MLOps) workflow for predicting customer churn. It includes data generation, preprocessing, model training, versioning, API deployment, performance monitoring, and model rollback functionality.

## 🏗️ Project Structure

```
.
├── data/                   # Data storage
│   ├── processed.csv      # Cleaned and preprocessed data
│   └── raw.csv            # Original raw data
├── logs/                   # Application logs
│   └── predictions.log    # Prediction request logs
├── models/                 # Saved model artifacts
│   └── churn_model_v*.joblib
├── registry/               # Model registry and metadata
│   ├── current_model.txt  # Active model reference
│   ├── metadata.json      # Model version metadata
│   └── train_stats.json   # Training statistics
└── src/                    # Source code
    ├── api.py             # FastAPI prediction service
    ├── evaluate.py        # Model evaluation utilities
    ├── generate_data.py   # Synthetic data generation
    ├── monitor_drift.py   # Data/model drift detection
    ├── prepare_data.py    # Data preprocessing pipeline
    ├── rollback.py        # Model rollback functionality
    └── train.py           # Model training pipeline
```

## 🚀 Features

- **Automated Data Pipeline**: Generate synthetic data and preprocess features
- **Model Versioning**: Track multiple model versions with metadata
- **Model Registry**: Centralized registry for model management
- **REST API**: FastAPI-based prediction service
- **Performance Monitoring**: Log predictions and monitor model performance
- **Drift Detection**: Monitor for data and model drift
- **Model Rollback**: Revert to previous model versions if needed
- **Comprehensive Logging**: Track all predictions and system events

## 🛠️ Technology Stack

- **ML Framework**: scikit-learn
- **API Framework**: FastAPI
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib
- **Logging**: Python logging module

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Create virtual environment
conda create -n <my-env> python=3.10
conda activate <my-env>

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Generate Data

```bash
python src/generate_data.py
```

### 2. Preprocess Data

```bash
python src/prepare_data.py
```

### 3. Train Model

```bash
python src/train.py
```

![Alt text](images/train-mlops-01.png)

### 4. Evaluate Model

```bash
python src/evaluate.py
```

![img.png](images/img.png)

### 5. Start Prediction API

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://localhost:8000`

### 6. Monitor Drift

```bash
python src/monitor_drift.py
```

![Alt text](images/drift_ss.png)

### 7. Rollback Model (if needed)

```bash
python src/rollback.py

#or towards a specific version using
python -c "from src.rollback import main; main('churn_model_v1_YYYYMMDD_HHMMSS.joblib')"
```

![img_2.png](images/img_2.png)

## 🔌 API Endpoints

### Health Check
```bash
GET /health

#example
curl -X GET http://127.0.0.1:8000/health 
```

![img_1.png](images/img_1.png)

### Make Prediction
```bash
POST /predict
Content-Type: application/json

{
  "tenure_months": 200,
  "num_complaints": 50,
  "avg_session_minutes": 500,
  "plan_type": "string",
  "region": "string",
  "request_id": "string"
}

#example request
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "tenure_months": 48,
  "num_complaints": 0,
  "avg_session_minutes": 60,
  "plan_type": "premium",
  "region": "EU",
  "request_id": "req-safe"
}'
``` 

## 📊 Model Registry

The model registry maintains:
- **current_model.txt**: Reference to the active production model
- **metadata.json**: Version history, timestamps, and model lineage
- **train_stats.json**: Performance metrics for each trained model

## 🔍 Monitoring

- **Predictions**: Logged to `logs/predictions.log`
- **Drift Detection**: Monitors feature distributions and model performance
- **Performance Metrics**: Tracked in `registry/train_stats.json`

## 🔄 Model Lifecycle

1. **Training**: New models are trained and saved with version timestamps
2. **Evaluation**: Models are evaluated against validation data
3. **Registration**: Model metadata and statistics are recorded
4. **Deployment**: Best performing model is set as current
5. **Monitoring**: Production model is monitored for drift
6. **Rollback**: Can revert to previous versions if performance degrades



# TP3

![tp3-pic1](tp3-pictures/tp3-pic1.png)

![tp3-pic2](tp3-pictures/tp3-pic2.png)

![tp3-pic3](tp3-pictures/tp3-pic3.png)

![tp3-pic4](tp3-pictures/tp3-pic4.png)

# TP 4

![tp4-pic1](tp4-pictures/tp4-pic1.png)

![tp4-pic2](tp4-pictures/tp4-pic2.png)


# TP 5

![tp5-pic1](tp5-pictures/Screenshot%20from%202026-01-06%2023-17-30.png)

![tp5-pic2](tp5-pictures/Screenshot%20from%202026-01-06%2023-18-12.png)

![tp5-pic3](tp5-pictures/Screenshot%20from%202026-01-06%2023-21-42.png)

![tp5-pic4](tp5-pictures/Screenshot%20from%202026-01-06%2023-23-57.png)

![tp5-pic5](tp5-pictures/Screenshot%20from%202026-01-06%2023-25-41.png)

![tp5-pic6](tp5-pictures/Screenshot%20from%202026-01-06%2023-30-18.png)

![tp5-pic7](tp5-pictures/Screenshot%20from%202026-01-06%2023-34-43.png)

![tp5-pic8](tp5-pictures/Screenshot%20from%202026-01-06%2023-37-40.png)

![tp5-pic9](tp5-pictures/Screenshot%20from%202026-01-06%2023-39-14.png)

![tp5-pic10](tp5-pictures/Screenshot%20from%202026-01-06%2023-39-31.png)

![tp5-pic11](tp5-pictures/Screenshot%20from%202026-01-06%2023-42-14.png)

![tp5-pic12](tp5-pictures/Screenshot%20from%202026-01-06%2023-46-14.png)

![tp5-pic13](tp5-pictures/Screenshot%20from%202026-01-06%2023-46-30.png)

![tp5-pic14](tp5-pictures/Screenshot%20from%202026-01-06%2023-46-42.png)

![tp5-pic15](tp5-pictures/Screenshot%20from%202026-01-06%2023-48-37.png)

![tp5-pic16](tp5-pictures/Screenshot%20from%202026-01-06%2023-49-15.png)

![tp5-pic17](tp5-pictures/Screenshot%20from%202026-01-06%2023-49-50.png)

![tp5-pic18](tp5-pictures/Screenshot%20from%202026-01-06%2023-50-26.png)

![tp5-pic19](tp5-pictures/Screenshot%20from%202026-01-06%2023-51-21.png)


## TP6

![Screenshot from 2026-01-21 21-10-37](tp6-pictures/Screenshot%20from%202026-01-21%2021-10-37.png)
![Screenshot from 2026-01-21 21-30-38](tp6-pictures/Screenshot%20from%202026-01-21%2021-30-38.png)
![Screenshot from 2026-01-21 21-40-15](tp6-pictures/Screenshot%20from%202026-01-21%2021-40-15.png)
![Screenshot from 2026-01-21 21-46-41](tp6-pictures/Screenshot%20from%202026-01-21%2021-46-41.png)
![Screenshot from 2026-01-21 21-57-05](tp6-pictures/Screenshot%20from%202026-01-21%2021-57-05.png)
![Screenshot from 2026-01-21 22-00-26](tp6-pictures/Screenshot%20from%202026-01-21%2022-00-26.png)
![Screenshot from 2026-01-21 22-09-42](tp6-pictures/Screenshot%20from%202026-01-21%2022-09-42.png)
![Screenshot from 2026-01-21 22-10-03](tp6-pictures/Screenshot%20from%202026-01-21%2022-10-03.png)
![Screenshot from 2026-01-21 22-19-07](tp6-pictures/Screenshot%20from%202026-01-21%2022-19-07.png)
![Screenshot from 2026-01-21 22-29-09](tp6-pictures/Screenshot%20from%202026-01-21%2022-29-09.png)
![Screenshot from 2026-01-21 22-29-16](tp6-pictures/Screenshot%20from%202026-01-21%2022-29-16.png)
![Screenshot from 2026-01-21 22-42-02](tp6-pictures/Screenshot%20from%202026-01-21%2022-42-02.png)
![Screenshot from 2026-01-22 01-24-25](tp6-pictures/Screenshot%20from%202026-01-22%2001-24-25.png)
![Screenshot from 2026-01-22 01-33-38](tp6-pictures/Screenshot%20from%202026-01-22%2001-33-38.png)
![Screenshot from 2026-01-22 01-36-57](tp6-pictures/Screenshot%20from%202026-01-22%2001-36-57.png)
![Screenshot from 2026-01-22 01-37-12](tp6-pictures/Screenshot%20from%202026-01-22%2001-37-12.png)
![Screenshot from 2026-01-22 01-43-03](tp6-pictures/Screenshot%20from%202026-01-22%2001-43-03.png)
![Screenshot from 2026-01-22 01-43-18](tp6-pictures/Screenshot%20from%202026-01-22%2001-43-18.png)
![Screenshot from 2026-01-22 01-45-27](tp6-pictures/Screenshot%20from%202026-01-22%2001-45-27.png)
![Screenshot from 2026-01-22 02-05-41](tp6-pictures/Screenshot%20from%202026-01-22%2002-05-41.png)
![Screenshot from 2026-01-22 02-06-11](tp6-pictures/Screenshot%20from%202026-01-22%2002-06-11.png)
![Screenshot from 2026-01-22 02-11-54](tp6-pictures/Screenshot%20from%202026-01-22%2002-11-54.png)
![Screenshot from 2026-01-22 02-16-30](tp6-pictures/Screenshot%20from%202026-01-22%2002-16-30.png)
![Screenshot from 2026-01-22 02-30-46](tp6-pictures/Screenshot%20from%202026-01-22%2002-30-46.png)
![Screenshot from 2026-01-22 03-27-13](tp6-pictures/Screenshot%20from%202026-01-22%2003-27-13.png)
![Screenshot from 2026-01-22 03-27-27](tp6-pictures/Screenshot%20from%202026-01-22%2003-27-27.png)
![Screenshot from 2026-01-22 03-27-40](tp6-pictures/Screenshot%20from%202026-01-22%2003-27-40.png)
![Screenshot from 2026-01-22 03-28-20](tp6-pictures/Screenshot%20from%202026-01-22%2003-28-20.png)
![Screenshot from 2026-01-22 03-28-59](tp6-pictures/Screenshot%20from%202026-01-22%2003-28-59.png)
![Screenshot from 2026-01-22 03-30-27](tp6-pictures/Screenshot%20from%202026-01-22%2003-30-27.png)
![Screenshot from 2026-01-22 03-31-53](tp6-pictures/Screenshot%20from%202026-01-22%2003-31-53.png)

## TP7

![Screenshot from 2026-01-22 04-33-21](tp7-pictures/Screenshot%20from%202026-01-22%2004-33-21.png)
![Screenshot from 2026-01-22 04-34-14](tp7-pictures/Screenshot%20from%202026-01-22%2004-34-14.png)
![Screenshot from 2026-01-22 04-36-06](tp7-pictures/Screenshot%20from%202026-01-22%2004-36-06.png)
![Screenshot from 2026-01-22 04-36-14](tp7-pictures/Screenshot%20from%202026-01-22%2004-36-14.png)
![Screenshot from 2026-01-22 04-57-06](tp7-pictures/Screenshot%20from%202026-01-22%2004-57-06.png)
![Screenshot from 2026-01-22 05-04-20](tp7-pictures/Screenshot%20from%202026-01-22%2005-04-20.png)
![Screenshot from 2026-01-22 05-05-03](tp7-pictures/Screenshot%20from%202026-01-22%2005-05-03.png)
![Screenshot from 2026-01-22 05-06-09](tp7-pictures/Screenshot%20from%202026-01-22%2005-06-09.png)
![Screenshot from 2026-01-22 05-06-30](tp7-pictures/Screenshot%20from%202026-01-22%2005-06-30.png)
![Screenshot from 2026-01-22 05-08-56](tp7-pictures/Screenshot%20from%202026-01-22%2005-08-56.png)
![Screenshot from 2026-01-22 05-09-14](tp7-pictures/Screenshot%20from%202026-01-22%2005-09-14.png)
![Screenshot from 2026-01-22 05-13-19](tp7-pictures/Screenshot%20from%202026-01-22%2005-13-19.png)


## 👥 Author

Boulaamail Mohamed ali



