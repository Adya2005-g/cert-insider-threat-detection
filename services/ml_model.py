from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier

from utils.constants import MODEL_FEATURES


# ------------------------------
# Model Path Setup
# ------------------------------
MODEL_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


# ------------------------------
# Train Models
# ------------------------------
def train_model(
    data: pd.DataFrame, 
    algorithm: str = "isolation_forest",
    contamination: float = 0.1
):
    """Train a selected model and persist it to disk."""
    training_frame = data[MODEL_FEATURES].astype(float)

    if algorithm == "isolation_forest":
        model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=42,
        )
    elif algorithm == "one_class_svm":
        model = OneClassSVM(
            nu=contamination,
            kernel="rbf",
            gamma="auto"
        )
    else:
        # Default to Isolation Forest if unknown
        model = IsolationForest(contamination=contamination, random_state=42)

    model.fit(training_frame)
    joblib.dump(model, MODEL_PATH)
    return model


# ------------------------------
# Load Model
# ------------------------------
def load_model() -> IsolationForest | None:
    """Load a persisted model if one is available."""
    if not os.path.exists(MODEL_PATH):
        print("No saved model found")
        return None

    return joblib.load(MODEL_PATH)


# ------------------------------
# Get or Train Model
# ------------------------------
def get_or_train_model(data: pd.DataFrame, algorithm: str = "isolation_forest") -> any:
    """Return an existing model or train one using the supplied dataset."""
    model = load_model()

    if model is None:
        print("Training new model...")
        return train_model(data, algorithm=algorithm)

    return model


# ------------------------------
# Detect Anomalies
# ------------------------------
def detect_anomalies(data: pd.DataFrame, retrain: bool = False) -> pd.DataFrame:
    """Run anomaly detection and attach anomaly predictions to the frame."""

    # Train or load model
    model = train_model(data) if retrain else get_or_train_model(data)

    scoring_frame = data[MODEL_FEATURES].astype(float)

    # Predictions
    predictions = model.predict(scoring_frame)  # 1 = normal, -1 = anomaly
    decision_scores = model.decision_function(scoring_frame)

    result = data.copy()

    # Convert to user-friendly format
    result["anomaly_prediction"] = predictions
    result["anomaly_flag"] = (predictions == -1).astype(int)  # 1 = threat
    result["anomaly_score"] = -decision_scores  # higher = riskier

    result["detection"] = result["anomaly_flag"].map({
        1: "Threat",
        0: "Normal"
    })

    return result


# ------------------------------
# Risk Score Calculation
# ------------------------------
def calculate_risk_score(data: pd.DataFrame) -> pd.Series:
    """Calculate a bounded risk score (0-100) using weights for various behaviors."""

    # Normalizing weights (safely handling missing columns)
    login_comp = np.clip(data.get("login_frequency", pd.Series(0, index=data.index)) / 20.0, 0, 1) * 10
    night_comp = np.clip(data.get("night_login_count", pd.Series(0, index=data.index)) / 2.0, 0, 1) * 25
    after_hours_comp = np.clip(data.get("after_hours_activity", pd.Series(0, index=data.index)) / 5.0, 0, 1) * 15
    file_comp = np.clip(data.get("file_access_count", pd.Series(0, index=data.index)) / 30.0, 0, 1) * 20
    email_comp = np.clip(data.get("email_activity_count", pd.Series(0, index=data.index)) / 50.0, 0, 1) * 15
    usb_comp = np.clip(data.get("usb_usage_count", pd.Series(0, index=data.index)) / 1.0, 0, 1) * 15

    # Base anomaly score influence
    anomaly_bonus = np.clip(data.get("anomaly_score", 0) * 50, 0, 10)

    risk_score = (
        login_comp
        + night_comp
        + after_hours_comp
        + file_comp
        + email_comp
        + usb_comp
        + anomaly_bonus
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
