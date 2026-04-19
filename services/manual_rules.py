from __future__ import annotations


def _to_number(value) -> float:
    try:
        numeric_value = float(value or 0)
    except (TypeError, ValueError):
        raise ValueError("All manual detection inputs must be numeric.")

    if numeric_value < 0:
        raise ValueError("Manual detection inputs cannot be negative.")
    return numeric_value


def normalize_manual_input(form_data: dict) -> dict:
    return {
        "login_frequency": _to_number(form_data.get("login_frequency", 0)),
        "after_hours_count": _to_number(form_data.get("after_hours_count", 0)),
        "night_login_count": _to_number(form_data.get("night_login_count", 0)),
        "file_access_count": _to_number(form_data.get("file_access_count", 0)),
        "email_activity_count": _to_number(form_data.get("email_activity_count", 0)),
        "usb_usage_count": _to_number(form_data.get("usb_usage_count", 0)),
    }


def evaluate_manual_behavior(form_data: dict) -> dict:
    metrics = normalize_manual_input(form_data)
    triggered_rules: list[str] = []
    risk_score = 0

    if metrics["login_frequency"] >= 4:
        triggered_rules.append("High Login Frequency")
        risk_score += 20

    if metrics["after_hours_count"] >= 2:
        triggered_rules.append("Repeated After-Hours Activity")
        risk_score += 20

    if metrics["night_login_count"] >= 1:
        triggered_rules.append("Multiple Night Logins")
        risk_score += 15

    if metrics["file_access_count"] >= 25:
        triggered_rules.append("Excessive File Access")
        risk_score += 20

    if metrics["email_activity_count"] >= 20:
        triggered_rules.append("Heavy Email Activity")
        risk_score += 15

    if metrics["usb_usage_count"] >= 2:
        triggered_rules.append("USB Usage Detected")
        risk_score += 25

    if (
        metrics["after_hours_count"] >= 2
        and metrics["usb_usage_count"] >= 1
        and metrics["file_access_count"] >= 20
    ):
        triggered_rules.append("Possible Data Theft Attempt")
        risk_score += 30

    if metrics["login_frequency"] >= 4 and metrics["night_login_count"] >= 1:
        triggered_rules.append("Possible Account Misuse")
        risk_score += 25

    risk_score = min(int(risk_score), 100)

    if risk_score >= 70:
        threat_status = "Threat"
        severity = "Critical"
        color_class = "danger"
        recommended_action = "Immediate investigation required."
    elif risk_score >= 40:
        threat_status = "Suspicious"
        severity = "Medium"
        color_class = "warning"
        recommended_action = "Monitor user activity closely."
    else:
        threat_status = "Normal"
        severity = "Low"
        color_class = "success"
        recommended_action = "No immediate concern."

    return {
        "inputs": metrics,
        "threat_status": threat_status,
        "severity": severity,
        "risk_score": risk_score,
        "triggered_rules": triggered_rules,
        "recommended_action": recommended_action,
        "color_class": color_class,
    }
