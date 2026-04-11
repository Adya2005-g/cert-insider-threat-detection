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
    elif "activity" in data.columns:
        # Count rows where activity contains 'file' or 'doc'
        file_mask = data["activity"].astype(str).str.lower().str.contains("file|doc", na=False)
        file_access = data[file_mask].groupby("user_id").size().reset_index(name="file_access_count")
    else:
        file_access = pd.DataFrame({"user_id": data["user_id"].unique(), "file_access_count": 0})

    # --------------------------
    # AFTER-HOURS ACTIVITY (Normal vs Night)
    # --------------------------
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data["hour"] = data["timestamp"].dt.hour

        # Unusual Night Activity (12AM - 6AM)
        data["night_activity"] = data["hour"].apply(lambda x: 1 if (x >= 0 and x < 6) else 0)
        # General After Hours (before 9AM or after 6PM)
        data["after_hours"] = data["hour"].apply(lambda x: 1 if (x < 9 or x > 18) else 0)

        after_hours_feat = (
            data.groupby("user_id")["after_hours"]
            .sum()
            .reset_index(name="after_hours_activity")
        )
        night_activity_feat = (
            data.groupby("user_id")["night_activity"]
            .sum()
            .reset_index(name="night_login_count")
        )
    else:
        logging.warning("⚠️ 'timestamp' column not found")
        after_hours_feat = pd.DataFrame({"user_id": data["user_id"].unique(), "after_hours_activity": 0})
        night_activity_feat = pd.DataFrame({"user_id": data["user_id"].unique(), "night_login_count": 0})

    # --------------------------
    # EMAIL ACTIVITY
    # --------------------------
    if "email_count" in data.columns:
        email_feat = data.groupby("user_id")["email_count"].sum().reset_index(name="email_activity_count")
    elif "activity" in data.columns:
        email_mask = data["activity"].astype(str).str.lower().str.contains("email|message", na=False)
        email_feat = data[email_mask].groupby("user_id").size().reset_index(name="email_activity_count")
    else:
        email_feat = pd.DataFrame({"user_id": data["user_id"].unique(), "email_activity_count": 0})

    # --------------------------
    # USB USAGE
    # --------------------------
    if "usb_usage" in data.columns:
        usb_feat = data.groupby("user_id")["usb_usage"].sum().reset_index(name="usb_usage_count")
    elif "activity" in data.columns:
        usb_mask = data["activity"].astype(str).str.lower().str.contains("usb|device|connect", na=False)
        usb_feat = data[usb_mask].groupby("user_id").size().reset_index(name="usb_usage_count")
    else:
        usb_feat = pd.DataFrame({"user_id": data["user_id"].unique(), "usb_usage_count": 0})

    # --------------------------
    # MERGE ALL FEATURES
    # --------------------------
    features = login_freq.merge(file_access, on="user_id", how="left")
    features = features.merge(after_hours_feat, on="user_id", how="left")
    features = features.merge(night_activity_feat, on="user_id", how="left")
    features = features.merge(email_feat, on="user_id", how="left")
    features = features.merge(usb_feat, on="user_id", how="left")

    # --------------------------
    # Handle Missing Values
    # --------------------------
    features = features.fillna(0)

    logging.info("✅ Feature engineering completed")
    logging.info(f"📊 Feature shape: {features.shape}")

    return features