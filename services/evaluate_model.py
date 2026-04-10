from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)


# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate(model, X_test: pd.DataFrame, y_true: pd.Series | None = None):
    """
    Evaluate Isolation Forest model.

    - Converts IsolationForest output:
        -1 → 1 (anomaly)
         1 → 0 (normal)

    - Uses dummy labels if y_true is not provided

    Args:
        model: Trained IsolationForest model
        X_test (pd.DataFrame): Test features
        y_true (pd.Series, optional): Ground truth labels

    Returns:
        dict: Evaluation metrics
    """

    if model is None:
        raise ValueError("Model is not provided")

    if X_test is None or X_test.empty:
        raise ValueError("X_test is empty")

    # --------------------------
    # Model Predictions
    # --------------------------
    raw_preds = model.predict(X_test)

    # Convert: -1 → 1 (anomaly), 1 → 0 (normal)
    y_pred = np.where(raw_preds == -1, 1, 0)

    # --------------------------
    # Handle Missing Ground Truth
    # --------------------------
    if y_true is None:
        logging.warning("⚠️ No ground truth provided → using dummy labels (all normal)")
        y_true = np.zeros(len(y_pred))

    # Ensure correct format
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    # --------------------------
    # Evaluation Metrics
    # ------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # --------------------------
    # Print Results
    # --------------------------
    print("\n========== Model Evaluation ==========")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("======================================\n")

    logging.info("✅ Model evaluation completed")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }