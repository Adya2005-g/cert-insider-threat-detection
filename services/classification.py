from __future__ import annotations

import logging
import pandas as pd


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)


# ------------------------------
# Classification Function
# ------------------------------
def classify_threat(anomalies) -> pd.Series:
    """
    Classify threat level based on anomaly values.

    Args:
        anomalies (pd.Series | list): List or Series of anomaly values
                                      (1 = anomaly, 0 = normal)

    Returns:
        pd.Series: Threat classification labels
    """

    if anomalies is None:
        raise ValueError("Input anomalies cannot be None")

    # Convert to pandas Series if needed
    if not isinstance(anomalies, pd.Series):
        anomalies = pd.Series(anomalies)

    # Validate values
    if not anomalies.isin([0, 1]).all():
        raise ValueError("Anomalies must contain only 0 or 1 values")

    # Classification
    labels = anomalies.map({
        1: "High Risk",
        0: "Normal"
    })

    logging.info("✅ Threat classification completed")

    return labels