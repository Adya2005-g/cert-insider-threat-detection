from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils.constants import MODEL_FEATURES


# ------------------------------
# Model Path Setup
# ------------------------------
MODEL_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


# ------------------------------
# Train Model
# ------------------------------
def train_isolation_forest(
    data: pd.DataFrame, contamination: float = 0.1
) -> IsolationForest:
    """Train a new Isolation Forest model and persist it to disk."""

    # Select only required features
    training_frame = data[MODEL_FEATURES].astype(float)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
    )

    model.fit(training_frame)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")

    return model


# ------------------------------
# Load Model
# ------------------------------
def load_model() -> IsolationForest | None:
    """Load a persisted model if one is available."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ No saved model found")
        return None

    return joblib.load(MODEL_PATH)


# ------------------------------
# Get or Train Model
# ------------------------------
def get_or_train_model(data: pd.DataFrame) -> IsolationForest:
    """Return an existing model or train one using the supplied dataset."""
    model = load_model()

    if model is None:
        print("🔁 Training new model...")
        return train_isolation_forest(data)

    return model


# ------------------------------
# Detect Anomalies
# ------------------------------
def detect_anomalies(data: pd.DataFrame, retrain: bool = False) -> pd.DataFrame:
    """Run anomaly detection and attach anomaly predictions to the frame."""

    # Train or load model
    model = train_isolation_forest(data) if retrain else get_or_train_model(data)

    scoring_frame = data[MODEL_FEATURES].astype(float)

    # Predictions
    predictions = model.predict(scoring_frame)  # 1 = normal, -1 = anomaly
    decision_scores = model.decision_function(scoring_frame)

    result = data.copy()

    # Convert to user-friendly format
    result["anomaly_prediction"] = predictions
    result["anomaly_flag"] = (predictions == -1).astype(int)  # 1 = threat
    result["anomaly_score"] = -decision_scores  # higher = riskier

    # 👇 ADD THIS (IMPORTANT for your UI issue)
    result["detection"] = result["anomaly_flag"].map({
        1: "Threat",
        0: "Normal"
    })

    return result


# ------------------------------
# Risk Score Calculation
# ------------------------------
def calculate_risk_score(data: pd.DataFrame) -> pd.Series:
    """Calculate a bounded risk score using engineered behavioral features."""

    login_component = np.clip(data["login_frequency"] / 10.0, 0, 1) * 25
    after_hours_component = np.clip(data["after_hours_activity"], 0, 1) * 35
    file_component = np.clip(data["file_access_count"] / 20.0, 0, 1) * 25
    anomaly_component = np.clip(data["anomaly_score"] * 100.0, 0, 15)

    risk_score = (
        login_component
        + after_hours_component
        + file_component
        + anomaly_component
    )

    return np.clip(risk_score, 0, 100)


# ------------------------------
# Threat Classification
# ------------------------------
def classify_threat(data: pd.DataFrame) -> pd.Series:
    """Translate numeric risk scores into human-readable threat levels."""

    conditions = [
        data["risk_score"] >= 80,
        data["risk_score"] >= 60,
        data["risk_score"] >= 30,
    ]

    labels = ["critical", "high", "medium"]

    return pd.Series(
        np.select(conditions, labels, default="low"),
        index=data.index
    )