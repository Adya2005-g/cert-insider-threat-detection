ALERT_THRESHOLD = 70.0
MODEL_FEATURES = ["login_frequency", "after_hours_activity", "night_login_count", "file_access_count", "email_activity_count", "usb_usage_count"]
REQUIRED_OUTPUT_COLUMNS = ["user_id", "anomaly_flag", "threat_level", "risk_score"]
CATEGORICAL_COLUMNS = ["activity", "action", "department", "device", "file_name", "email_subject"]
