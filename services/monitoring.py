from __future__ import annotations

import logging
import time
import pandas as pd


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ------------------------------
# Monitoring Function
# ------------------------------
def monitor(data: pd.DataFrame, delay: float = 1.0):
    """
    Simulate real-time monitoring of user activity.

    Args:
        data (pd.DataFrame): Input dataset
        delay (float): Time delay between records (seconds)

    Returns:
        None
    """

    if data is None or data.empty:
        raise ValueError("Input data is empty")

    logging.info("🚀 Starting real-time monitoring...")

    try:
        for index, row in data.iterrows():
            user = row.get("user_id", "Unknown")
            activity = row.to_dict()

            # Log activity
            logging.info(f"👤 User: {user} | Activity: {activity}")

            # Simulate real-time delay
            time.sleep(delay)

        logging.info("✅ Monitoring completed")

    except KeyboardInterrupt:
        logging.warning("⏹️ Monitoring stopped by user")

    except Exception as e:
        logging.error(f"❌ Monitoring error: {str(e)}")
        raise