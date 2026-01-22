from __future__ import annotations

"""
Module d'entraînement et d'enregistrement d'un modèle de churn.

Pipeline MLOps minimal :
- entraînement
- évaluation
- gate qualité
- enregistrement modèle + métadonnées
- tracking MLflow
"""

from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Final
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = ROOT / "data" / "processed.csv"
MODELS_DIR: Final[Path] = ROOT / "models"
REGISTRY_DIR: Final[Path] = ROOT / "registry"

CURRENT_MODEL_PATH: Final[Path] = REGISTRY_DIR / "current_model.txt"
METADATA_PATH: Final[Path] = REGISTRY_DIR / "metadata.json"

MODEL_NAME: Final[str] = "churn_model"

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def load_metadata() -> list[dict[str, Any]]:
    if not METADATA_PATH.exists():
        return []

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(items: list[dict[str, Any]]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_baseline_f1(y_true: pd.Series | list[int]) -> float:
    y_pred = [0] * len(y_true)
    return float(f1_score(y_true, y_pred, zero_division=0))


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_model_pipeline(
    preprocessor: ColumnTransformer,
    seed: int,
) -> Pipeline:
    classifier = LogisticRegression(
        max_iter=200,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", classifier),
        ]
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main(version: str = "v1", seed: int = 42, gate_f1: float = 0.65) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Fichier processed.csv introuvable. "
            "Veuillez exécuter la préparation des données."
        )

    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "churn"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    numeric_cols = ["tenure_months", "num_complaints", "avg_session_minutes"]
    categorical_cols = ["plan_type", "region"]

    preprocessor = build_preprocessing_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    model_pipeline = build_model_pipeline(
        preprocessor=preprocessor,
        seed=seed,
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    # Train
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = model_pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "baseline_f1": compute_baseline_f1(y_test),
    }

    # Save model locally
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    model_filename = f"{MODEL_NAME}_{version}_{timestamp}.joblib"
    model_path = MODELS_DIR / model_filename

    joblib.dump(model_pipeline, model_path)

    # -----------------------------------------------------------------------
    # MLflow tracking
    # -----------------------------------------------------------------------

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("mlops-lab-01")

    with mlflow.start_run(run_name=f"train-{version}") as run:
        mlflow.log_param("version", version)
        mlflow.log_param("seed", seed)
        mlflow.log_param("gate_f1", gate_f1)

        mlflow.log_metrics(metrics)

        mlflow.set_tag("data_file", DATA_PATH.name)
        mlflow.set_tag("model_file", model_filename)

        mlflow.log_artifact(
            str(model_path),
            artifact_path="exported_models",
        )

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

    # -----------------------------------------------------------------------
    # Registry metadata
    # -----------------------------------------------------------------------

    entry: dict[str, Any] = {
        "model_file": model_filename,
        "version": version,
        "trained_at_utc": timestamp,
        "data_file": DATA_PATH.name,
        "seed": seed,
        "metrics": metrics,
        "gate_f1": gate_f1,
        "passed_gate": bool(
            metrics["f1"] >= gate_f1
            and metrics["f1"] >= metrics["baseline_f1"]
        ),
    }

    items = load_metadata()
    items.append(entry)
    save_metadata(items)

    print("[METRICS]", json.dumps(metrics, indent=2))
    print(f"[OK] Modèle sauvegardé : {model_path}")

    if entry["passed_gate"]:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        CURRENT_MODEL_PATH.write_text(model_filename, encoding="utf-8")
        print(f"[DEPLOY] Modèle activé : {model_filename}")
    else:
        print("[DEPLOY] Refusé par le gate qualité")


if __name__ == "__main__":
    main()
