from models.alert import Alert


def generate_alerts(logs, threshold: float):
    """Create alert records for logs that exceed the configured threshold."""
    alerts = []
    for log in logs:
        score = log.risk_score.score if log.risk_score else 0.0
        if score > threshold:
            alerts.append(
                Alert(
                    log_id=log.id,
                    risk_score=score,
                    threshold=threshold,
                    severity="critical" if score >= 85 else "high",
                    message=f"High-risk insider activity detected for user {log.user_id}.",
                )
            )
    return alerts
