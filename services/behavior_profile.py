from __future__ import annotations

import logging
import pandas as pd


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)


# ------------------------------
# Create User Behavior Profile
# ------------------------------
def create_user_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user behavior profile by aggregating average behavior features.

    Args:
        df (pd.DataFrame): Feature-engineered dataframe

    Returns:
        pd.DataFrame: User behavior profile
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if "user_id" not in df.columns:
        raise ValueError("Column 'user_id' is required")

    data = df.copy()

    # --------------------------
    # Select numeric columns only
    # --------------------------
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

    # Remove user_id from aggregation
    if "user_id" in numeric_cols:
        numeric_cols.remove("user_id")

    if not numeric_cols:
        raise ValueError("No numeric columns found for profiling")

    # --------------------------
    # Group by user and calculate mean
    # --------------------------
    profile_df = (
        data.groupby("user_id")[numeric_cols]
        .mean()
        .reset_index()
    )

    # --------------------------
    # Rename columns (optional clarity)
    # --------------------------
    profile_df = profile_df.rename(
        columns={col: f"avg_{col}" for col in numeric_cols}
    )

    logging.info("✅ User behavior profiles created successfully")
    logging.info(f"📊 Profile shape: {profile_df.shape}")

    return profile_df