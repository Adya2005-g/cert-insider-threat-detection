import sys
import os

# Add the project root to sys.path
sys.path.append(r'b:\work\cert-insider-threat-detection')

try:
    from services.ml_model import detect_anomalies
    print("Successfully imported detect_anomalies")
except Exception as e:
    print(f"Error importing or running: {e}")
    sys.exit(1)
