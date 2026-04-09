import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from services.data_pipeline import feature_engineering, preprocess
from services.ml_model import train_isolation_forest


def train_from_csv(csv_path: str):
    """Utility entry point for training from a CSV file on disk."""

    # ------------------------------
    # 1. Load Dataset
    # ------------------------------
    dataset = pd.read_csv(csv_path)

    if dataset.empty:
        raise Exception("Dataset is empty")

    # ------------------------------
    # 2. Check Label Column
    # ------------------------------
    if "label" not in dataset.columns:
        raise Exception("Dataset must contain a 'label' column (0 = normal, 1 = threat)")

    y = dataset["label"]
    X = dataset.drop(columns=["label"])

    # ------------------------------
    # 3. Preprocessing
    # ------------------------------
    processed = preprocess(X)

    # ------------------------------
    # 4. Feature Engineering
    # ------------------------------
    features = feature_engineering(processed)

    # ------------------------------
    # 5. Train-Test Split
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )

    # ------------------------------
    # 6. Train Model (Isolation Forest)
    # ------------------------------
    model = train_isolation_forest(X_train)

    # ------------------------------
    # 7. Prediction
    # ------------------------------
    # IsolationForest: 1 = normal, -1 = anomaly
    y_pred = model.predict(X_test)

    # Convert to 0 (normal), 1 (threat)
    y_pred = np.where(y_pred == -1, 1, 0)

    # ------------------------------
    # 8. Evaluation
    # ------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("\n========== Model Evaluation ==========")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("======================================\n")

    # ------------------------------
    # 9. Save Model
    # ------------------------------
    joblib.dump(model, "model.pkl")
    print("✅ Model saved as model.pkl")

    return model


# ------------------------------
# MAIN ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    train_from_csv("uploads/sample.csv")