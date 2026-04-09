from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from utils.constants import CATEGORICAL_COLUMNS, MODEL_FEATURES, REQUIRED_OUTPUT_COLUMNS


def load_csv_dataset(file_storage) -> pd.DataFrame:
    """Load a CSV dataset uploaded through Flask."""
    return pd.read_csv(file_storage)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [column.strip().lower().replace(" ", "_") for column in renamed.columns]
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
    renamed = renamed.rename(columns={k: v for k, v in aliases.items() if k in renamed.columns})
    return renamed


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw CERT-like logs for downstream ML processing."""
    frame = _normalize_columns(data)

    if "user_id" not in frame.columns:
        frame["user_id"] = [f"user_{index + 1}" for index in range(len(frame))]

    if "timestamp" not in frame.columns:
        frame["timestamp"] = pd.Timestamp.utcnow()

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["timestamp"] = frame["timestamp"].fillna(pd.Timestamp.utcnow())

    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].fillna("unknown")
        else:
            frame[column] = frame[column].fillna(0)

    for column in CATEGORICAL_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("category").cat.codes

    if "file_access_count" not in frame.columns:
        file_related_columns = [column for column in frame.columns if "file" in column and column != "user_id"]
        if file_related_columns:
            frame["file_access_count"] = frame[file_related_columns].select_dtypes(include=["number"]).sum(axis=1)
        else:
            frame["file_access_count"] = 0

    frame["event_hour"] = frame["timestamp"].dt.hour.astype(int)
    frame["event_day"] = frame["timestamp"].dt.dayofweek.astype(int)

    return frame


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Create behavioral features used by the anomaly detector."""
    frame = data.copy()

    if not is_datetime64_any_dtype(frame["timestamp"]):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)

    frame["login_frequency"] = frame.groupby("user_id")["timestamp"].transform("count").astype(float)
    frame["after_hours_activity"] = frame["event_hour"].apply(lambda hour: 1.0 if hour < 6 or hour >= 20 else 0.0)
    frame["file_access_count"] = pd.to_numeric(frame["file_access_count"], errors="coerce").fillna(0).astype(float)

    for column in MODEL_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0

    return frame


def format_output_records(data: pd.DataFrame) -> list[dict]:
    """Build the API response shape required by the project."""
    return data[REQUIRED_OUTPUT_COLUMNS].to_dict(orient="records")
