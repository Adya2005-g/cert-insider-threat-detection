from __future__ import annotations

import logging
import pandas as pd


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)


# ------------------------------
# Feature Engineering Function
# ------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML-ready features:
    - login_frequency per user
    - after_hours_activity (before 9AM or after 6PM)
    - file_access_count

    Args:
        df (pd.DataFrame): Preprocessed dataframe

    Returns:
        pd.DataFrame: Feature dataframe ready for ML
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    data = df.copy()

    # --------------------------
    # Check Required Column
    # --------------------------
    if "user_id" not in data.columns:
        raise ValueError("Column 'user_id' is required")

    # --------------------------
    # LOGIN FREQUENCY
    # --------------------------
    login_freq = data.groupby("user_id").size().reset_index(name="login_frequency")

    # --------------------------
    # FILE ACCESS COUNT
    # --------------------------
    if "file_access_count" in data.columns:
        file_access = data.groupby("user_id")["file_access_count"].sum().reset_index()
    else:
        file_access = data.groupby("user_id").size().reset_index(name="file_access_count")

    # --------------------------
    # AFTER-HOURS ACTIVITY
    # --------------------------
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data["hour"] = data["timestamp"].dt.hour

        data["after_hours"] = data["hour"].apply(
            lambda x: 1 if (x < 9 or x > 18) else 0
        )

        after_hours = (
            data.groupby("user_id")["after_hours"]
            .sum()
            .reset_index(name="after_hours_activity")
        )
    else:
        logging.warning("⚠️ 'timestamp' column not found → after_hours set to 0")
        after_hours = data.groupby("user_id").size().reset_index()
        after_hours["after_hours_activity"] = 0
        after_hours = after_hours[["user_id", "after_hours_activity"]]

    # --------------------------
    # MERGE ALL FEATURES
    # --------------------------
    features = login_freq.merge(file_access, on="user_id", how="left")
    features = features.merge(after_hours, on="user_id", how="left")

    # --------------------------
    # Handle Missing Values
    # --------------------------
    features = features.fillna(0)

    logging.info("✅ Feature engineering completed")
    logging.info(f"📊 Feature shape: {features.shape}")

    return features