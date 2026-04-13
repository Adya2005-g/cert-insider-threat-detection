import pandas as pd
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(r'b:\work\cert-insider-threat-detection')

from services.data_pipeline import feature_engineering
from services.ml_model import calculate_risk_score

# Dummy data
data = {
    "user_id": ["user1", "user1", "user2"],
    "timestamp": ["2026-04-13 02:00:00", "2026-04-13 14:00:00", "2026-04-13 10:00:00"],
    "file_access_count": [10, 5, 2]
}

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("--- Running Feature Engineering ---")
engineered = feature_engineering(df)
print("Columns in engineered DataFrame:", engineered.columns.tolist())

# Check for night_login_count
if "night_login_count" in engineered.columns:
    print("night_login_count column exists.")
    print(engineered[["user_id", "timestamp", "night_login_count"]])
else:
    print("Error: night_login_count column MISSING.")

print("\n--- Running Risk Scoring ---")
# Mock anomaly_score
engineered["anomaly_score"] = 0.5
scores = calculate_risk_score(engineered)
print("Calculated Risk Scores:")
print(scores)

print("\nVerification Complete.")
