import os
import re

import pandas as pd
from sklearn.ensemble import IsolationForest


REQUIRED_EMAIL_COLUMNS = {
    "id",
    "date",
    "user",
    "pc",
    "to",
    "cc",
    "bcc",
    "from",
    "size",
    "attachments",
    "content",
}

SENSITIVE_KEYWORDS = [
    "password",
    "confidential",
    "salary",
    "finance",
    "client_data",
    "secret",
]


def is_email_dataset(df: pd.DataFrame) -> bool:
    return REQUIRED_EMAIL_COLUMNS.issubset({str(column).strip().lower() for column in df.columns})


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(column).strip().lower() for column in renamed.columns]
    return renamed


def _split_recipients(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    parts = re.split(r"[;,]+", text)
    return [part.strip().lower() for part in parts if part.strip()]


def _extract_domain(address: str) -> str:
    if not address or "@" not in address:
        return ""
    return address.rsplit("@", 1)[-1].strip().lower()


def _infer_company_domain(df: pd.DataFrame) -> str:
    configured = os.environ.get("COMPANY_DOMAIN", "").strip().lower()
    if configured:
        return configured

    sender_domains = (
        df["from"]
        .fillna("")
        .astype(str)
        .str.lower()
        .map(_extract_domain)
    )
    valid_domains = sender_domains[sender_domains != ""]
    if valid_domains.empty:
        return ""
    return str(valid_domains.mode().iloc[0])


def _calculate_email_frequency(df: pd.DataFrame) -> pd.Series:
    prepared = df[["user", "date"]].copy()
    prepared["row_index"] = prepared.index
    prepared = prepared.sort_values(["user", "date"])

    frequencies = pd.Series(1.0, index=prepared.index, dtype=float)
    for user, group in prepared.groupby("user"):
        rolling_counts = (
            group.set_index("date")
            .rolling("15min")
            .count()["row_index"]
            .astype(float)
        )
        frequencies.loc[group.index] = rolling_counts.values

    return frequencies.sort_index()


def analyze_email_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    email_df = _normalize_columns(df)
    if not REQUIRED_EMAIL_COLUMNS.issubset(email_df.columns):
        missing = sorted(REQUIRED_EMAIL_COLUMNS.difference(email_df.columns))
        raise ValueError(f"Email dataset is missing required columns: {', '.join(missing)}")

    email_df["date"] = pd.to_datetime(email_df["date"], errors="coerce")
    email_df["user"] = email_df["user"].fillna("unknown").astype(str).str.strip().str.lower()
    email_df["attachments"] = pd.to_numeric(email_df["attachments"], errors="coerce").fillna(0).clip(lower=0)
    email_df["size"] = pd.to_numeric(email_df["size"], errors="coerce").fillna(0).clip(lower=0)
    email_df["content"] = email_df["content"].fillna("").astype(str)
    email_df = email_df.dropna(subset=["date"]).copy()

    if email_df.empty:
        raise ValueError("The uploaded email dataset does not contain any valid date values.")

    company_domain = _infer_company_domain(email_df)
    email_df["hour"] = email_df["date"].dt.hour.astype(int)
    email_df["weekday"] = email_df["date"].dt.weekday.astype(int)
    email_df["is_after_hours"] = (email_df["hour"] >= 21) | (email_df["hour"] < 6)
    email_df["is_weekend"] = email_df["weekday"] >= 5

    recipient_lists = []
    recipient_counts = []
    external_flags = []
    for _, row in email_df.iterrows():
        to_list = _split_recipients(row["to"])
        cc_list = _split_recipients(row["cc"])
        bcc_list = _split_recipients(row["bcc"])
        recipients = to_list + cc_list + bcc_list
        recipient_lists.append(recipients)
        recipient_counts.append(len(recipients))

        recipient_domains = {_extract_domain(address) for address in recipients if _extract_domain(address)}
        has_external = bool(company_domain and any(domain != company_domain for domain in recipient_domains))
        external_flags.append(has_external)

    email_df["recipient_list"] = recipient_lists
    email_df["recipient_count"] = recipient_counts
    email_df["external_email_detected"] = external_flags
    email_df["email_frequency"] = _calculate_email_frequency(email_df)
    email_df["large_attachment_sent"] = (email_df["attachments"] > 0) & (email_df["size"] > 30000)
    email_df["too_many_attachments"] = email_df["attachments"] >= 3
    email_df["mass_recipients"] = email_df["recipient_count"] > 5
    email_df["burst_email_activity"] = email_df["email_frequency"] > 10
    email_df["sensitive_content_detected"] = email_df["content"].str.contains(
        "|".join(re.escape(keyword) for keyword in SENSITIVE_KEYWORDS),
        case=False,
        na=False,
    )

    features = email_df[["hour", "weekday", "attachments", "size", "recipient_count", "email_frequency"]].astype(float)
    contamination = float(
        min(
            max(
                (
                    email_df[
                        [
                            "is_after_hours",
                            "is_weekend",
                            "large_attachment_sent",
                            "too_many_attachments",
                            "mass_recipients",
                            "external_email_detected",
                            "burst_email_activity",
                            "sensitive_content_detected",
                        ]
                    ].any(axis=1).mean()
                    or 0.05
                ),
                0.05,
            ),
            0.25,
        )
    )
    model = IsolationForest(contamination=contamination, random_state=42)
    email_df["ml_prediction"] = model.fit_predict(features)
    email_df["ml_anomaly_score"] = model.decision_function(features)

    triggered_rules = []
    for _, row in email_df.iterrows():
        row_rules = []
        if row["is_after_hours"]:
            row_rules.append("Late Night Email Sent")
        if row["is_weekend"]:
            row_rules.append("Weekend Email Activity")
        if row["large_attachment_sent"]:
            row_rules.append("Large Attachment Sent")
        if row["too_many_attachments"]:
            row_rules.append("Multiple Attachments Sent")
        if row["mass_recipients"]:
            row_rules.append("Mass Email Distribution")
        if row["external_email_detected"]:
            row_rules.append("External Email Sharing")
        if row["burst_email_activity"]:
            row_rules.append("Unusual High Email Frequency")
        if row["sensitive_content_detected"]:
            row_rules.append("Sensitive Content Detected")
        if row["ml_prediction"] == -1:
            row_rules.append("ML Anomaly Pattern")
        triggered_rules.append(row_rules)

    email_df["triggered_rules"] = triggered_rules
    email_df["triggered_rule"] = email_df["triggered_rules"].map(lambda rules: ", ".join(rules) if rules else "No suspicious activity")
    email_df["threat_status"] = email_df["triggered_rules"].map(lambda rules: "Threat" if rules else "Normal")
    email_df["severity"] = email_df["triggered_rules"].map(lambda rules: "Critical" if rules else "Low")

    email_df["risk_score"] = (
        email_df["triggered_rules"].map(len) * 12
        + (email_df["ml_prediction"] == -1).astype(int) * 10
        + email_df["attachments"].clip(upper=5) * 2
        + (email_df["size"] / 5000).clip(upper=20)
    ).clip(0, 100).round(2)

    email_df["source_type"] = "email_insider"
    email_df["company_domain"] = company_domain
    email_df["sender_domain"] = email_df["from"].fillna("").astype(str).str.lower().map(_extract_domain)
    return email_df, company_domain
