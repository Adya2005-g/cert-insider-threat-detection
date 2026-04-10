from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)


# ------------------------------
# Train Model
# ------------------------------
def train_model(X: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
    """
    Train Isolation Forest model.

    Args:
        X (pd.DataFrame): Feature dataset
        contamination (float): Expected anomaly proportion

    Returns:
        IsolationForest: Trained model
    """

    if X is None or X.empty:
        raise ValueError("Training data is empty")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )

    model.fit(X)

    logging.info("✅ Isolation Forest model trained successfully")

    return model


# ------------------------------
# Detect Anomalies
# ------------------------------
def detect_anomalies(model: IsolationForest, X: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies using trained model.

    Args:
        model (IsolationForest): Trained model
        X (pd.DataFrame): Feature dataset

    Returns:
        pd.DataFrame: DataFrame with anomaly predictions
    """

    if model is None:
        raise ValueError("Model is not provided")

    if X is None or X.empty:
        raise ValueError("Input data is empty")

    # IsolationForest output: 1 = normal, -1 = anomaly
    raw_preds = model.predict(X)

    # Convert to required format: 1 = anomaly, 0 = normal
    anomaly_flags = np.where(raw_preds == -1, 1, 0)

    result = X.copy()
    result["anomaly"] = anomaly_flags

    logging.info("✅ Anomaly detection completed")
    logging.info(f"📊 Total anomalies detected: {anomaly_flags.sum()}")

    return result