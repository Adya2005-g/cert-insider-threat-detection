from __future__ import annotations

import logging
import pandas as pd


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)


# ------------------------------
# Risk Scoring Function
# ------------------------------
def calculate_risk(anomalies) -> list[float]:
    """
    Assign risk scores based on anomaly values.

    Rules:
        1 → 0.9 (High Risk)
        0 → 0.1 (Normal)

    Args:
        anomalies (list | pd.Series): List or Series of anomaly values

    Returns:
        list[float]: Risk scores
    """

    if anomalies is None:
        raise ValueError("Input anomalies cannot be None")

    # Convert to pandas Series if needed
    if not isinstance(anomalies, pd.Series):
        anomalies = pd.Series(anomalies)

    # Validate input values
    if not anomalies.isin([0, 1]).all():
        raise ValueError("Anomalies must contain only 0 or 1 values")

    # Apply scoring
    risk_scores = anomalies.map({
        1: 0.9,
        0: 0.1
    })

    logging.info("✅ Risk scoring completed")

    return risk_scores.tolist()