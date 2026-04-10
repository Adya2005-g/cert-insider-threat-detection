from __future__ import annotations

import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

# Import your modules
from services.data_loader import load_data
from services.preprocessing import preprocess
from services.feature_engineering import feature_engineering
from services.anomaly_detection import train_model


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ------------------------------
# Select Numeric Features
# ------------------------------
def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only numeric columns for training."""
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        raise ValueError("❌ No numeric features found")

    return numeric_df


# ------------------------------
# Training Pipeline
# ------------------------------
def run_training_pipeline(
    file_path: str,
    model_path: str = "model.pkl"
):
    """
    Full training pipeline:
    - Load data
    - Preprocess
    - Feature engineering
    - Select numeric features
    - Train/test split
    - Train model
    - Save model
    """

    try:
        # --------------------------
        # 1. Load Data
        # --------------------------
        df = load_data(file_path)
        logging.info("📥 Data loaded successfully")

        # --------------------------
        # 2. Preprocessing
        # --------------------------
        df = preprocess(df)
        logging.info("🧹 Preprocessing completed")

        # --------------------------
        # 3. Feature Engineering
        # --------------------------
        features = feature_engineering(df)
        logging.info("⚙️ Feature engineering completed")

        # --------------------------
        # 4. Select Numeric Features
        # --------------------------
        X = select_numeric_features(features)

        # --------------------------
        # 5. Train-Test Split
        # --------------------------
        X_train, X_test = train_test_split(
            X,
            test_size=0.2,
            random_state=42
        )

        logging.info(f"📊 Train shape: {X_train.shape}")
        logging.info(f"📊 Test shape: {X_test.shape}")

        # --------------------------
        # 6. Train Model
        # --------------------------
        model = train_model(X_train)

        # --------------------------
        # 7. Save Model
        # --------------------------
        joblib.dump(model, model_path)
        logging.info(f"✅ Model saved at: {model_path}")

        return model

    except Exception as e:
        logging.error(f"❌ Training failed: {str(e)}")
        raise


# ------------------------------
# Main Entry Point
# ------------------------------
if __name__ == "__main__":
    # Change file path if needed
    run_training_pipeline("uploads/sample.csv")