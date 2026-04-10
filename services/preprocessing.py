from __future__ import annotations

import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.constants import CATEGORICAL_COLUMNS


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)


# ------------------------------
# Preprocess Function
# ------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preprocessing on dataset:
    - Handle missing values
    - Convert timestamp to datetime
    - Extract hour feature
    - Encode categorical columns

    Args:
        df (pd.DataFrame): Raw input dataframe

    Returns:
        pd.DataFrame: Processed dataframe
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    data = df.copy()

    # --------------------------
    # Handle Missing Values
    # --------------------------
    data = data.fillna(0)
    logging.info("✅ Missing values handled")

    # --------------------------
    # Timestamp Processing
    # --------------------------
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

        # Extract hour
        data["hour"] = data["timestamp"].dt.hour.fillna(0)

        logging.info("✅ Timestamp converted and hour extracted")

    else:
        logging.warning("⚠️ 'timestamp' column not found")

    # --------------------------
    # Encode Categorical Columns
    # --------------------------
    label_encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    if label_encoders:
        logging.info(f"✅ Encoded categorical columns: {list(label_encoders.keys())}")
    else:
        logging.info("ℹ️ No categorical columns found to encode")

    # --------------------------
    # Final Cleanup
    # --------------------------
    data = data.fillna(0)

    return data