from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from utils.constants import CATEGORICAL_COLUMNS, MODEL_FEATURES, REQUIRED_OUTPUT_COLUMNS


# ------------------------------
# Load CSV (for Flask uploads)
# ------------------------------
def load_csv_dataset(file_storage) -> pd.DataFrame:
    """Load a CSV dataset uploaded through Flask."""
    return pd.read_csv(file_storage)


# ------------------------------
# Normalize Column Names
# ------------------------------
def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and map aliases."""

    renamed = frame.copy()

    # Normalize column names safely
    renamed.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in renamed.columns
    ]

    # Column alias mapping
    aliases = {
        "user": "user_id",
        "employee_id": "user_id",
        "employee": "user_id",
        "userid": "user_id",

        "date": "timestamp",
        "datetime": "timestamp",
        "activity_time": "timestamp",
        "time": "timestamp",

        "file_count": "file_access_count",
        "files_accessed": "file_access_count",
        "file_access": "file_access_count",
    }

    # Apply alias mapping
    renamed = renamed.rename(
        columns={k: v for k, v in aliases.items() if k in renamed.columns}
    )

    return renamed


# ------------------------------
# Preprocessing
# ------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare dataset."""

    df = _normalize_columns(df)

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(0)

    return df


# ------------------------------
# Feature Engineering
# ------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features while preserving analyst-facing columns."""

    df = df.copy()

    # --------------------------
    # Timestamp Handling
    # --------------------------
    if "timestamp" in df.columns:
        if not is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        df["hour"] = df["timestamp"].dt.hour.fillna(0)

        # After-hours activity (before 9 or after 18)
        df["after_hours_activity"] = df["hour"].apply(
            lambda x: 1 if x < 9 or x > 18 else 0
        )

        # Night login activity (between 0 and 5 AM)
        df["is_night"] = df["hour"].apply(
            lambda x: 1 if 0 <= x <= 5 else 0
        )

    # --------------------------
    # Aggregate Features per User
    # --------------------------
    if "user_id" in df.columns:
        # Calculate frequencies if the columns don't exist yet
        if "login_frequency" not in df.columns:
            df["login_frequency"] = df.groupby("user_id")["user_id"].transform("count")
        
        if "night_login_count" not in df.columns and "is_night" in df.columns:
            df["night_login_count"] = df.groupby("user_id")["is_night"].transform("sum")

    # --------------------------
    # Default Feature Handling & Initialization
    # --------------------------
    for feature in MODEL_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
        # Replace NaN with 0 for all model features
        df[feature] = df[feature].fillna(0).astype(float)

    return df


# ------------------------------
# Final Output Preparation (Optional for UI)
# ------------------------------
def prepare_output(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required output columns exist for frontend."""

    df = df.copy()

    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df
