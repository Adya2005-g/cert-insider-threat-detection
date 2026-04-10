from __future__ import annotations

import logging
import pandas as pd


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)


# ------------------------------
# Merge Logs Function
# ------------------------------
def merge_logs(
    login_df: pd.DataFrame,
    email_df: pd.DataFrame,
    file_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge multiple log dataframes (login, email, file logs)
    using 'user_id' as the key.

    Args:
        login_df (pd.DataFrame): Login activity logs
        email_df (pd.DataFrame): Email activity logs
        file_df (pd.DataFrame): File access logs

    Returns:
        pd.DataFrame: Combined dataframe
    """

    # --------------------------
    # Input Validation
    # --------------------------
    for name, df in {
        "login_df": login_df,
        "email_df": email_df,
        "file_df": file_df,
    }.items():
        if df is None or df.empty:
            logging.warning(f"⚠️ {name} is empty or None")

        if df is not None and "user_id" not in df.columns:
            raise ValueError(f"{name} must contain 'user_id' column")

    # --------------------------
    # Ensure DataFrames are valid
    # --------------------------
    login_df = login_df.copy() if login_df is not None else pd.DataFrame()
    email_df = email_df.copy() if email_df is not None else pd.DataFrame()
    file_df = file_df.copy() if file_df is not None else pd.DataFrame()

    # --------------------------
    # Merge Logs (Outer Join)
    # --------------------------
    try:
        merged_df = pd.merge(
            login_df,
            email_df,
            on="user_id",
            how="outer",
            suffixes=("_login", "_email")
        )

        merged_df = pd.merge(
            merged_df,
            file_df,
            on="user_id",
            how="outer"
        )

        logging.info("✅ Logs merged successfully")

    except Exception as e:
        logging.error(f"❌ Error during merging: {str(e)}")
        raise

    # --------------------------
    # Handle Missing Values
    # --------------------------
    merged_df = merged_df.fillna(0)

    logging.info(f"📊 Final merged shape: {merged_df.shape}")

    return merged_df