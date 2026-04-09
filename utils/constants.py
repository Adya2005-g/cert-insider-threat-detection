ALERT_THRESHOLD = 70.0
MODEL_FEATURES = ["login_frequency", "after_hours_activity", "file_access_count"]
REQUIRED_OUTPUT_COLUMNS = ["user_id", "anomaly_flag", "threat_level", "risk_score"]
CATEGORICAL_COLUMNS = ["activity", "action", "department", "device", "file_name", "email_subject"]
