from __future__ import annotations

from io import BytesIO, StringIO

import pandas as pd


def _serialize_log_row(log) -> dict:
    raw_record = log.raw_record or {}
    username = log.user.username if log.user else str(log.user_id)
    score = float(log.risk_score.score) if log.risk_score else 0.0

    if raw_record.get("source_type") == "email_insider":
        return {
            "user": username,
            "date": raw_record.get("date", ""),
            "attachments": raw_record.get("attachments", 0),
            "size": raw_record.get("size", 0),
            "threat_status": raw_record.get("threat_status", "Threat" if log.anomaly_flag else "Normal"),
            "severity": raw_record.get("severity", str(log.threat_level).title()),
            "triggered_rule": raw_record.get("triggered_rule", "No suspicious activity"),
            "recipient_count": raw_record.get("recipient_count", 0),
            "email_frequency": raw_record.get("email_frequency", 0),
            "ml_prediction": raw_record.get("ml_prediction", ""),
            "risk_score": score,
        }

    return {
        "user": username,
        "event_timestamp": log.event_timestamp.isoformat() if log.event_timestamp else "",
        "login_frequency": log.login_frequency,
        "after_hours_activity": log.after_hours_activity,
        "night_login_count": log.night_login_count,
        "file_access_count": log.file_access_count,
        "email_activity_count": log.email_activity_count,
        "usb_usage_count": log.usb_usage_count,
        "threat_status": "Threat" if log.anomaly_flag else "Normal",
        "severity": str(log.threat_level).title(),
        "risk_score": score,
    }


def _build_dataframe(logs: list) -> tuple[pd.DataFrame, str]:
    source_type = (logs[0].raw_record or {}).get("source_type", "") if logs else ""
    rows = [_serialize_log_row(log) for log in logs]
    return pd.DataFrame(rows), source_type


def _build_summary_dataframe(df: pd.DataFrame, source_type: str, batch_id: str) -> pd.DataFrame:
    if source_type == "email_insider":
        threat_count = int((df["threat_status"] == "Threat").sum()) if not df.empty else 0
        normal_count = int((df["threat_status"] == "Normal").sum()) if not df.empty else 0
        critical_count = int((df["severity"] == "Critical").sum()) if not df.empty else 0
        summary = {
            "Batch ID": batch_id,
            "Dataset Type": "Email Insider Detection",
            "Total Emails": len(df),
            "Threat Emails": threat_count,
            "Normal Emails": normal_count,
            "Critical Alerts": critical_count,
            "Average Risk Score": round(float(df["risk_score"].mean()), 2) if not df.empty else 0.0,
        }
    else:
        threat_count = int((df["threat_status"] == "Threat").sum()) if not df.empty else 0
        summary = {
            "Batch ID": batch_id,
            "Dataset Type": "Behavior Detection",
            "Records Processed": len(df),
            "Threat Records": threat_count,
            "Normal Records": len(df) - threat_count,
            "Average Risk Score": round(float(df["risk_score"].mean()), 2) if not df.empty else 0.0,
        }

    return pd.DataFrame([summary])


def export_batch_report(logs: list, batch_id: str, export_format: str) -> tuple[BytesIO, str, str]:
    df, source_type = _build_dataframe(logs)
    summary_df = _build_summary_dataframe(df, source_type, batch_id)
    prefix = "email_insider_report" if source_type == "email_insider" else "report"

    if export_format == "csv":
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        byte_buffer = BytesIO(csv_buffer.getvalue().encode("utf-8"))
        byte_buffer.seek(0)
        return byte_buffer, f"{prefix}_{batch_id}.csv", "text/csv"

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        df.to_excel(writer, index=False, sheet_name="Detailed Report")
    excel_buffer.seek(0)
    return (
        excel_buffer,
        f"{prefix}_{batch_id}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
