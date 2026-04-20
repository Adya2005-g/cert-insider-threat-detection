from datetime import datetime, timedelta
from email.message import EmailMessage
from functools import wraps
import logging
import os
import secrets
import smtplib
from uuid import uuid4

import joblib
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import generate_password_hash

from alerts.manager import generate_alerts
from models.alert import Alert
from models.log import Log
from models.risk_score import RiskScore
from models.user import User
from routes.api import api_bp
from routes.manual_detect import manual_detect_bp
from routes.report_export import report_export_bp
from services.anomaly_detection import detect_anomalies as detect_pipeline_anomalies
from services.data_pipeline import feature_engineering as batch_feature_engineering
from services.data_pipeline import load_csv_dataset
from services.data_pipeline import preprocess as batch_preprocess
from services.ml_model import calculate_risk_score, classify_threat, detect_anomalies
from services.behavior_profile import create_user_profile
from services.classification import classify_threat as classify_pipeline_threat
from services.data_loader import load_data
from services.evaluate_model import evaluate
from services.email_insider_detection import analyze_email_dataset, is_email_dataset
from services.feature_engineering import feature_engineering
from services.log_integration import merge_logs
from services.monitoring import monitor
from services.preprocessing import preprocess
from services.risk_scoring import calculate_risk
from services.auth_service import authenticate_user
from utils.constants import ALERT_THRESHOLD
from utils.config import DevelopmentConfig
from utils.constants import MODEL_FEATURES
from utils.extensions import db
from utils.seed import seed_default_user


logging.basicConfig(level=logging.INFO)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
_pipeline_model = None


def _send_reset_code_email(recipient: str, code: str) -> bool:
    smtp_host = os.environ.get("MAIL_SERVER")
    smtp_port = int(os.environ.get("MAIL_PORT", "587"))
    smtp_username = os.environ.get("MAIL_USERNAME")
    smtp_password = os.environ.get("MAIL_PASSWORD")
    sender = os.environ.get("MAIL_DEFAULT_SENDER", smtp_username or "no-reply@example.com")
    use_tls = os.environ.get("MAIL_USE_TLS", "true").lower() == "true"

    if not smtp_host or not sender:
        return False

    message = EmailMessage()
    message["Subject"] = "Your InsiderDetection password reset code"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(
        f"Use this verification code to reset your password: {code}\n\n"
        "This code will expire in 10 minutes."
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            if use_tls:
                server.starttls()
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            server.send_message(message)
        return True
    except Exception as exc:
        logging.error("Failed to send password reset email: %s", exc)
        return False


def _clear_reset_session():
    session.pop("reset_email", None)
    session.pop("reset_code", None)
    session.pop("reset_code_expires_at", None)


def _set_reset_session(email: str, code: str):
    session["reset_email"] = email
    session["reset_code"] = code
    session["reset_code_expires_at"] = (datetime.utcnow() + timedelta(minutes=10)).isoformat()


def _reset_session_valid(email: str, code: str) -> bool:
    stored_email = session.get("reset_email")
    stored_code = session.get("reset_code")
    expires_at = session.get("reset_code_expires_at")

    if not stored_email or not stored_code or not expires_at:
        return False

    try:
        expiry = datetime.fromisoformat(expires_at)
    except ValueError:
        _clear_reset_session()
        return False

    if datetime.utcnow() > expiry:
        _clear_reset_session()
        return False

    return stored_email == email and stored_code == code


def _build_username(first_name: str, last_name: str, email: str) -> str:
    base_username = f"{first_name}.{last_name}".strip(".").lower().replace(" ", "")
    if not base_username:
        base_username = email.split("@", 1)[0].lower()

    candidate = base_username
    counter = 1
    while User.query.filter_by(username=candidate).first():
        counter += 1
        candidate = f"{base_username}{counter}"
    return candidate


def _current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def _login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if _current_user() is None:
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)

    return wrapped_view


def load_model():
    global _pipeline_model

    if _pipeline_model is not None:
        return _pipeline_model

    try:
        _pipeline_model = joblib.load(MODEL_PATH)
        logging.info("Model loaded successfully from %s", MODEL_PATH)
        return _pipeline_model
    except Exception as exc:
        logging.error("Model load failed: %s", exc)
        raise


def _prepare_model_frame(features: pd.DataFrame) -> pd.DataFrame:
    scoring_columns = [column for column in MODEL_FEATURES if column in features.columns]
    if len(scoring_columns) != len(MODEL_FEATURES):
        missing = [column for column in MODEL_FEATURES if column not in features.columns]
        raise ValueError(f"Missing model feature columns: {', '.join(missing)}")

    return features[scoring_columns].astype(float)


def _load_uploaded_csv(file_storage) -> pd.DataFrame | None:
    if file_storage is None or file_storage.filename == "":
        return None

    return pd.read_csv(file_storage)


def run_full_pipeline(file_path: str):
    dataset = load_data(file_path)
    processed = preprocess(dataset)
    features = feature_engineering(processed)
    profile = create_user_profile(features)

    scoring_frame = _prepare_model_frame(features)
    anomaly_result = detect_pipeline_anomalies(load_model(), scoring_frame)

    result = features.copy()
    result["anomaly"] = anomaly_result["anomaly"]
    result["threat_level"] = classify_pipeline_threat(result["anomaly"])
    result["risk_score"] = calculate_risk(result["anomaly"])

    monitor(result.head(3), delay=0.2)

    return result, profile, features


def _dashboard_modules():
    return [
        {"name": "Data Acquisition", "icon": "fa-database", "status": "Ready", "detail": "Collect CERT-style login, email, file, and device activity through upload or API ingestion."},
        {"name": "Preprocessing", "icon": "fa-filter", "status": "Ready", "detail": "Normalize timestamps, encode categories, and clean missing values before scoring."},
        {"name": "Log Integration", "icon": "fa-diagram-project", "status": "Ready", "detail": "Merge multiple event streams into one user-centric activity timeline."},
        {"name": "Feature Engineering", "icon": "fa-gears", "status": "Ready", "detail": "Generate login frequency, after-hours activity, file access, and derived behavior signals."},
        {"name": "Behavior Profiling", "icon": "fa-id-badge", "status": "Ready", "detail": "Build each user's normal usage baseline for comparison against current behavior."},
        {"name": "Anomaly Detection", "icon": "fa-magnifying-glass-chart", "status": "Ready", "detail": "Use ML models to flag unusual activity patterns and outlier behavior."},
        {"name": "Threat Classification", "icon": "fa-shield-virus", "status": "Ready", "detail": "Label activity as normal, suspicious, or high risk based on model outputs."},
        {"name": "Risk Scoring", "icon": "fa-gauge-high", "status": "Ready", "detail": "Calculate user risk severity scores to prioritize analyst review."},
        {"name": "Model Evaluation", "icon": "fa-vial-circle-check", "status": "Ready", "detail": "Measure precision, recall, and accuracy for the detection pipeline."},
        {"name": "Monitoring", "icon": "fa-satellite-dish", "status": "Ready", "detail": "Support near real-time review of newly processed user activity records."},
        {"name": "API & Access Control", "icon": "fa-key", "status": "Ready", "detail": "Expose secured Flask routes for upload, detection, risk retrieval, and authentication."},
        {"name": "Alerts & Reports", "icon": "fa-file-shield", "status": "Ready", "detail": "Generate analyst-facing alerts and downloadable investigation summaries."},
    ]


def _serialize_dashboard_alert(alert: Alert) -> dict:
    user = alert.log.user if alert.log else None
    return {
        "user": user.username if user else f"user-{alert.log.user_id if alert.log else 'unknown'}",
        "severity": alert.severity.title(),
        "message": alert.message,
        "risk_score": round(float(alert.risk_score), 2),
        "created_at": alert.created_at.strftime("%Y-%m-%d %H:%M") if alert.created_at else "",
    }


def _serialize_result_row(log: Log) -> dict:
    user = log.user.username if log.user else str(log.user_id)
    score = log.risk_score.score if log.risk_score else 0.0
    raw_record = log.raw_record or {}
    if raw_record.get("source_type") == "email_insider":
        return {
            "user": user,
            "date": raw_record.get("date", ""),
            "attachments": int(float(raw_record.get("attachments", 0) or 0)),
            "size": round(float(raw_record.get("size", 0) or 0), 2),
            "threat_status": raw_record.get("threat_status", "Threat" if log.anomaly_flag else "Normal"),
            "severity": raw_record.get("severity", str(log.threat_level).title()),
            "triggered_rule": raw_record.get("triggered_rule", "No suspicious activity"),
            "risk_score": round(float(score), 2),
            "status": "Threat" if log.anomaly_flag else "Normal",
            "threat_level": str(log.threat_level).title(),
            "source_type": "email_insider",
        }
    return {
        "user": user,
        "login_frequency": round(float(log.login_frequency), 2),
        "after_hours_activity": round(float(log.after_hours_activity), 2),
        "night_login_count": round(float(log.night_login_count), 2),
        "file_access_count": round(float(log.file_access_count), 2),
        "email_activity_count": round(float(log.email_activity_count), 2),
        "usb_usage_count": round(float(log.usb_usage_count), 2),
        "threat_level": str(log.threat_level).title(),
        "risk_score": round(float(score), 2),
        "status": "Threat" if log.anomaly_flag else "Normal",
    }


def _build_email_results_payload(logs: list[Log]) -> dict:
    rows = [_serialize_result_row(log) for log in logs]
    threat_rows = [row for row in rows if row.get("threat_status", row.get("status")) == "Threat"]
    normal_rows = [row for row in rows if row.get("threat_status", row.get("status")) == "Normal"]
    critical_rows = [row for row in rows if row.get("severity") == "Critical"]

    user_threat_counts: dict[str, int] = {}
    hourly_activity = {str(hour).zfill(2): 0 for hour in range(24)}
    attachment_analysis = {"Attachment Threats": 0, "Non-Attachment Threats": 0}

    for log in logs:
        raw = log.raw_record or {}
        if raw.get("threat_status") != "Threat":
            continue

        user_label = log.user.username if log.user else str(log.user_id)
        user_threat_counts[user_label] = user_threat_counts.get(user_label, 0) + 1

        hour_value = raw.get("hour")
        if hour_value is not None:
            try:
                hourly_activity[str(int(hour_value)).zfill(2)] += 1
            except (TypeError, ValueError):
                pass

        try:
            if float(raw.get("attachments", 0) or 0) > 0:
                attachment_analysis["Attachment Threats"] += 1
            else:
                attachment_analysis["Non-Attachment Threats"] += 1
        except (TypeError, ValueError):
            attachment_analysis["Non-Attachment Threats"] += 1

    summary = {
        "Total Emails": len(rows),
        "Threat Emails": len(threat_rows),
        "Normal Emails": len(normal_rows),
        "Critical Alerts": len(critical_rows),
    }

    return {
        "summary": summary,
        "rows": rows,
        "charts": {
            "threat_vs_normal": {
                "Threat": len(threat_rows),
                "Normal": len(normal_rows),
            },
            "user_threat_count": user_threat_counts,
            "hourly_suspicious_activity": hourly_activity,
            "attachment_threat_analysis": attachment_analysis,
        },
    }


def _latest_batch_logs():
    latest_log = Log.query.order_by(Log.created_at.desc(), Log.id.desc()).first()
    if latest_log is None:
        return None, []

    batch_logs = (
        Log.query.filter_by(batch_id=latest_log.batch_id)
        .order_by(Log.id.asc())
        .all()
    )
    return latest_log.batch_id, batch_logs


def _dashboard_context():
    batch_id, batch_logs = _latest_batch_logs()
    recent_alerts = (
        Alert.query.order_by(Alert.created_at.desc(), Alert.id.desc())
        .limit(5)
        .all()
    )
    all_logs_count = Log.query.count()
    anomalies_count = Log.query.filter_by(anomaly_flag=True).count()
    user_count = User.query.filter(User.role != "analyst").count()

    avg_risk = (
        db.session.query(db.func.avg(RiskScore.score)).scalar()
        if all_logs_count
        else None
    )

    latest_rows = [_serialize_result_row(log) for log in batch_logs[:6]]
    high_risk_count = sum(
        1 for log in batch_logs
        if log.risk_score and float(log.risk_score.score) >= ALERT_THRESHOLD
    )

    if avg_risk is None:
        avg_risk_label = "N/A"
        risk_note = "Upload a dataset to generate risk metrics."
    elif avg_risk >= 80:
        avg_risk_label = "Critical"
        risk_note = f"Average risk score {avg_risk:.1f}"
    elif avg_risk >= 60:
        avg_risk_label = "High"
        risk_note = f"Average risk score {avg_risk:.1f}"
    elif avg_risk >= 30:
        avg_risk_label = "Medium"
        risk_note = f"Average risk score {avg_risk:.1f}"
    else:
        avg_risk_label = "Low"
        risk_note = f"Average risk score {avg_risk:.1f}"

    # Chart Data: Threat Levels
    threat_distribution = {
        "Normal": all_logs_count - anomalies_count,
        "Threats": anomalies_count
    }

    # Chart Data: Risk Buckets
    risk_buckets = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for log in batch_logs:
        score = float(log.risk_score.score) if log.risk_score else 0
        if score >= 80: risk_buckets["Critical"] += 1
        elif score >= 60: risk_buckets["High"] += 1
        elif score >= 30: risk_buckets["Medium"] += 1
        else: risk_buckets["Low"] += 1

    latest_batch_source_type = ""
    if batch_logs:
        latest_batch_source_type = (batch_logs[0].raw_record or {}).get("source_type", "")

    return {
        "stats": {
            "users_monitored": user_count,
            "active_anomalies": anomalies_count,
            "records_processed": all_logs_count,
            "risk_label": avg_risk_label,
            "risk_note": risk_note,
            "alerts_count": Alert.query.count(),
            "latest_batch_records": len(batch_logs),
            "high_risk_count": high_risk_count,
        },
        "charts": {
            "threat_distribution": threat_distribution,
            "risk_buckets": risk_buckets
        },
        "modules": _dashboard_modules(),
        "recent_alerts": [_serialize_dashboard_alert(alert) for alert in recent_alerts],
        "latest_batch_id": batch_id,
        "latest_results": latest_rows,
        "latest_batch_source_type": latest_batch_source_type,
    }


def _process_uploaded_dataset(file_storage):
    dataset = load_csv_dataset(file_storage)
    if is_email_dataset(dataset):
        return _process_email_uploaded_dataset(dataset, file_storage.filename)

    processed = batch_preprocess(dataset)
    engineered = batch_feature_engineering(processed)
    scored = detect_anomalies(engineered, retrain=True)
    scored["risk_score"] = calculate_risk_score(scored)
    scored["threat_level"] = classify_threat(scored)

    batch_id = uuid4().hex
    known_users = {user.username: user for user in User.query.all()}
    logs_to_create = []

    for _, row in scored.iterrows():
        username = str(row["user_id"]).strip().lower()
        user = known_users.get(username)
        if user is None:
            user = User(
                username=username,
                email=f"{username}@generated.local",
                password_hash=generate_password_hash("employee123"),
                role="employee",
            )
            db.session.add(user)
            db.session.flush()
            known_users[username] = user

        log = Log(
            batch_id=batch_id,
            user_id=user.id,
            event_timestamp=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else None,
            login_frequency=float(row["login_frequency"]),
            after_hours_activity=float(row["after_hours_activity"]),
            night_login_count=float(row["night_login_count"]),
            file_access_count=float(row["file_access_count"]),
            email_activity_count=float(row["email_activity_count"]),
            usb_usage_count=float(row["usb_usage_count"]),
            anomaly_flag=bool(row["anomaly_flag"]),
            threat_level=str(row["threat_level"]),
            source_filename=file_storage.filename,
            raw_record={key: _json_safe_value(value) for key, value in row.to_dict().items()},
        )
        db.session.add(log)
        db.session.flush()

        risk_score = RiskScore(log=log, score=float(row["risk_score"]))
        db.session.add(risk_score)
        logs_to_create.append(log)

    db.session.flush()
    generated_alerts = generate_alerts(logs_to_create, ALERT_THRESHOLD)
    db.session.add_all(generated_alerts)
    db.session.commit()

    return batch_id


def _process_email_uploaded_dataset(dataset: pd.DataFrame, source_filename: str) -> str:
    analyzed, company_domain = analyze_email_dataset(dataset)
    batch_id = uuid4().hex
    known_users = {user.username: user for user in User.query.all()}
    logs_to_create: list[Log] = []
    alerts_to_create: list[Alert] = []

    for _, row in analyzed.iterrows():
        username = str(row["user"]).strip().lower()
        user = known_users.get(username)
        if user is None:
            user = User(
                username=username,
                email=f"{username}@{company_domain or 'generated.local'}",
                password_hash=generate_password_hash("employee123"),
                role="employee",
            )
            db.session.add(user)
            db.session.flush()
            known_users[username] = user

        rule_count = len(row["triggered_rules"])
        log = Log(
            batch_id=batch_id,
            user_id=user.id,
            event_timestamp=row["date"].to_pydatetime() if hasattr(row["date"], "to_pydatetime") else row["date"],
            login_frequency=0.0,
            after_hours_activity=1.0 if bool(row["is_after_hours"]) else 0.0,
            night_login_count=1.0 if bool(row["is_after_hours"]) else 0.0,
            file_access_count=0.0,
            email_activity_count=float(row["email_frequency"]),
            usb_usage_count=0.0,
            anomaly_flag=bool(row["threat_status"] == "Threat"),
            threat_level=str(row["severity"]).lower(),
            source_filename=source_filename,
            raw_record={
                "source_type": "email_insider",
                "email_id": _json_safe_value(row["id"]),
                "date": _json_safe_value(row["date"]),
                "user": username,
                "pc": _json_safe_value(row["pc"]),
                "to": _json_safe_value(row["to"]),
                "cc": _json_safe_value(row["cc"]),
                "bcc": _json_safe_value(row["bcc"]),
                "from": _json_safe_value(row["from"]),
                "size": _json_safe_value(row["size"]),
                "attachments": _json_safe_value(row["attachments"]),
                "content": _json_safe_value(row["content"]),
                "hour": _json_safe_value(row["hour"]),
                "weekday": _json_safe_value(row["weekday"]),
                "recipient_count": _json_safe_value(row["recipient_count"]),
                "email_frequency": _json_safe_value(row["email_frequency"]),
                "ml_prediction": _json_safe_value(row["ml_prediction"]),
                "ml_anomaly_score": _json_safe_value(row["ml_anomaly_score"]),
                "triggered_rule": _json_safe_value(row["triggered_rule"]),
                "triggered_rules": [_json_safe_value(value) for value in row["triggered_rules"]],
                "threat_status": _json_safe_value(row["threat_status"]),
                "severity": _json_safe_value(row["severity"]),
                "rule_count": rule_count,
                "company_domain": _json_safe_value(row["company_domain"]),
                "sender_domain": _json_safe_value(row["sender_domain"]),
            },
        )
        db.session.add(log)
        db.session.flush()

        risk_score = RiskScore(log=log, score=float(row["risk_score"]))
        db.session.add(risk_score)
        logs_to_create.append(log)

        if row["threat_status"] == "Threat":
            alerts_to_create.append(
                Alert(
                    log_id=log.id,
                    risk_score=float(row["risk_score"]),
                    threshold=ALERT_THRESHOLD,
                    severity="critical",
                    message=str(row["triggered_rule"]),
                )
            )

    db.session.flush()
    db.session.add_all(alerts_to_create)
    db.session.commit()
    return batch_id


def _json_safe_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def create_app(config_object=DevelopmentConfig):
    """Application factory used by Flask and tests."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    db.init_app(app)
    app.register_blueprint(api_bp, url_prefix="/api")
    from routes.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(manual_detect_bp)
    app.register_blueprint(report_export_bp)

    @app.context_processor
    def inject_common_context():
        user = _current_user()
        batch_id, _ = _latest_batch_logs()
        return {
            "current_user": user or User.anonymous(),
            "latest_batch_id": batch_id
        }

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def healthcheck():
        return jsonify(
            {
                "service": "Insider Threat Detection System",
                "status": "ok",
                "available_endpoints": [
                    "/login",
                    "/register",
                    "/dashboard",
                    "/detect?file=...",
                    "/evaluate?file=...",
                    "/merge",
                    "/api/login",
                    "/api/upload",
                    "/api/detect",
                    "/api/risk",
                ],
            }
        )

    @app.route("/login", methods=["GET", "POST"])
    def login_page():
        if request.method == "POST":
            identifier = request.form.get("identifier", "").strip()
            password = request.form.get("password", "")
            user = authenticate_user(identifier, password)

            if user is None:
                flash("Invalid username/email or password.", "error")
                return render_template("login.html")

            session["user_id"] = user.id
            flash("Signed in successfully.", "success")
            return redirect(url_for("dashboard"))

        return render_template("login.html")

    @app.route("/forgot-password", methods=["GET", "POST"])
    def forgot_password():
        # GET always starts at Step 1 (Request Email)
        # POST handle sequence (Request -> Verify -> Reset)
        step = "request"
        email_prefill = session.get("reset_email", "")

        if request.method == "POST":
            action = request.form.get("action", "request")

            if action == "request":
                email = request.form.get("email", "").strip().lower()
                user = User.query.filter_by(email=email).first()

                if not email:
                    flash("Please enter your registered email address.", "error")
                    return render_template("forgot_password.html", step="request", email=email)

                if user is None:
                    flash("No account was found with that email address.", "error")
                    return render_template("forgot_password.html", step="request", email=email)

                import secrets
                from services.mail_service import send_otp_email
                code = f"{secrets.randbelow(1000000):06d}"
                user.set_otp(code)
                db.session.commit()
                
                session["reset_email"] = email

                if send_otp_email(email, code):
                    flash("A verification code has been sent to your email.", "success")
                else:
                    flash(
                        f"Email delivery is not configured, so your demo verification code is {code}.",
                        "success",
                    )

                return render_template("forgot_password.html", step="verify", email=email)

            if action == "verify":
                email = request.form.get("email", "").strip().lower()
                code = request.form.get("code", "").strip()
                
                user = User.query.filter_by(email=email).first()
                if user and user.verify_otp(code):
                    session["reset_authorized"] = True
                    # IMPORTANT: Render Step 3 directly after successful verification
                    flash("Identity verified. Please set your new password.", "success")
                    return render_template("forgot_password.html", step="reset", email=email)
                else:
                    flash("Invalid or expired verification code.", "error")
                    return render_template("forgot_password.html", step="verify", email=email)

            if action == "reset":
                email = request.form.get("email", "").strip().lower()
                new_password = request.form.get("new_password", "")
                confirm_password = request.form.get("confirm_password", "")

                if not session.get("reset_authorized"):
                    flash("Session unauthorized. Please verify your OTP again.", "error")
                    return render_template("forgot_password.html", step="request", email=email)

                if new_password != confirm_password:
                    flash("Passwords must match.", "error")
                    return render_template("forgot_password.html", step="reset", email=email)

                user = User.query.filter_by(email=email).first()
                if not user:
                    flash("User not found.", "error")
                    return render_template("forgot_password.html", step="request", email=email)

                user.set_password(new_password)
                user.otp_hash = None
                user.otp_expiry = None
                db.session.commit()
                
                session.pop("reset_email", None)
                session.pop("reset_authorized", None)
                flash("Password reset successfully. Please login with your new credentials.", "success")
                return redirect(url_for("login_page"))

        return render_template("forgot_password.html", step=step, email=email_prefill)

    @app.route("/register", methods=["GET", "POST"])
    def register_page():
        if request.method == "POST":
            first_name = request.form.get("fname", "").strip()
            last_name = request.form.get("lname", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not all([first_name, last_name, email, password, confirm_password]):
                flash("Please fill in all required fields.", "error")
                return render_template("register.html")

            if password != confirm_password:
                flash("Passwords do not match.", "error")
                return render_template("register.html")

            if User.query.filter_by(email=email).first():
                flash("An account with this email already exists.", "error")
                return render_template("register.html")

            username = _build_username(first_name, last_name, email)
            user = User(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name,
                role="analyst",
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            session["user_id"] = user.id
            flash("Account created successfully.", "success")
            return redirect(url_for("dashboard"))

        return render_template("register.html")

    @app.route("/dashboard", methods=["GET"])
    @_login_required
    def dashboard():
        return render_template("dashboard.html", **_dashboard_context())

    @app.route("/profile", methods=["GET", "POST"])
    @_login_required
    def profile():
        user = _current_user()
        if request.method == "POST":
            first_name = request.form.get("first_name", "").strip()
            last_name = request.form.get("last_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            new_password = request.form.get("new_password", "").strip()
            
            # Simple updates
            if first_name: user.first_name = first_name
            if last_name: user.last_name = last_name
            
            if email and email != user.email:
                if User.query.filter_by(email=email).first():
                    flash("Email is already taken.", "error")
                else:
                    user.email = email
            
            if new_password:
                user.set_password(new_password)
            
            db.session.commit()
            flash("Profile updated successfully.", "success")
            return redirect(url_for("profile"))

        return render_template("profile.html", user=user)

    @app.route("/detection", methods=["GET"])
    @app.route("/detect", methods=["GET"])
    @_login_required
    def detect_view():
        file_path = request.args.get("file", "").strip()
        if file_path:
            try:
                result, profile, _ = run_full_pipeline(file_path)
                return jsonify(
                    {
                        "status": "success",
                        "data": result.to_dict(orient="records"),
                        "profile": profile.to_dict(orient="records"),
                    }
                )
            except Exception as exc:
                logging.error("Full pipeline detection failed: %s", exc)
                return jsonify({"error": str(exc)}), 500

        default_form = {
            "login_frequency": 0,
            "after_hours_count": 0,
            "night_login_count": 0,
            "file_access_count": 0,
            "email_activity_count": 0,
            "usb_usage_count": 0,
        }
        return render_template("behavior_profiling.html", analysis=None, form_data=default_form)

    @app.route("/detect", methods=["POST"])
    @_login_required
    def predict():
        try:
            login_frequency = float(request.form.get("login_frequency", 0))
            after_hours_activity = float(request.form.get("after_hours_activity", 0))
            night_login_count = float(request.form.get("night_login_count", 0))
            file_access_count = float(request.form.get("file_access_count", 0))
            email_activity_count = float(request.form.get("email_activity_count", 0))
            usb_usage_count = float(request.form.get("usb_usage_count", 0))
        except ValueError:
            flash("Enter numeric values for all analysis fields.", "error")
            return render_template("detection.html", analysis=None), 400

        analysis_input = pd.DataFrame(
            [
                {
                    "login_frequency": login_frequency,
                    "after_hours_activity": after_hours_activity,
                    "night_login_count": night_login_count,
                    "file_access_count": file_access_count,
                    "email_activity_count": email_activity_count,
                    "usb_usage_count": usb_usage_count,
                }
            ]
        )

        scored = detect_anomalies(analysis_input)
        scored["risk_score"] = calculate_risk_score(scored)
        scored["threat_level"] = classify_threat(scored)

        result = scored.iloc[0]
        analysis = {
            "detection": result.get("detection", "Unknown"),
            "risk_score": round(float(result["risk_score"]), 2),
            "threat_level": str(result["threat_level"]).title(),
            "anomaly_score": round(float(result["anomaly_score"]), 4),
            "login_frequency": login_frequency,
            "after_hours_activity": int(after_hours_activity),
            "file_access_count": file_access_count,
        }

        return render_template("detection.html", analysis=analysis)

    @app.route("/evaluate", methods=["GET"])
    @_login_required
    def evaluate_model():
        file_path = request.args.get("file", "").strip()
        if not file_path:
            return jsonify({"error": "File path required"}), 400

        try:
            dataset = load_data(file_path)
            processed = preprocess(dataset)
            features = feature_engineering(processed)
            metrics = evaluate(load_model(), _prepare_model_frame(features))
            return jsonify({"status": "success", "metrics": metrics})
        except Exception as exc:
            logging.error("Model evaluation failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/merge", methods=["POST"])
    @_login_required
    def merge():
        try:
            login_df = _load_uploaded_csv(request.files.get("login"))
            email_df = _load_uploaded_csv(request.files.get("email"))
            file_df = _load_uploaded_csv(request.files.get("file"))
            merged = merge_logs(login_df, email_df, file_df)
            return jsonify(
                {
                    "status": "success",
                    "data": merged.to_dict(orient="records"),
                }
            )
        except Exception as exc:
            logging.error("Log merge failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/upload", methods=["GET", "POST"])
    @_login_required
    def upload():
        if request.method == "POST":
            file = request.files.get("file")
            if file is None or file.filename == "":
                flash("Please choose a CSV file to upload.", "error")
                return render_template("upload.html")

            if not file.filename.lower().endswith(".csv"):
                flash("Only CSV files are supported.", "error")
                return render_template("upload.html")

            try:
                batch_id = _process_uploaded_dataset(file)
                flash("Dataset uploaded and analyzed successfully.", "success")
                return redirect(url_for("results", batch_id=batch_id))
            except Exception as exc:
                logging.error("Dataset upload failed: %s", exc)
                flash(str(exc), "error")

        return render_template("upload.html")

    @app.route("/results", methods=["GET"])
    @_login_required
    def results():
        batch_id = request.args.get("batch_id")
        if not batch_id:
            batch_id, _ = _latest_batch_logs()

        if not batch_id:
            return render_template("results.html", summary=None, rows=[], batch_id=None, charts={}, source_type="")

        logs = (
            Log.query.filter_by(batch_id=batch_id)
            .order_by(Log.id.asc())
            .all()
        )
        source_type = (logs[0].raw_record or {}).get("source_type", "") if logs else ""

        if source_type == "email_insider":
            payload = _build_email_results_payload(logs)
            return render_template(
                "results.html",
                summary=payload["summary"],
                rows=payload["rows"],
                charts=payload["charts"],
                batch_id=batch_id,
                source_type=source_type,
            )

        rows = [_serialize_result_row(log) for log in logs]
        summary = {
            "Records Processed": len(rows),
            "Threats Detected": sum(1 for log in logs if log.anomaly_flag),
            "Alerts Generated": sum(1 for log in logs if log.risk_score and log.risk_score.score >= ALERT_THRESHOLD),
            "Average Risk": round(
                sum(log.risk_score.score for log in logs if log.risk_score) / max(len(rows), 1),
                2,
            ),
        }

        return render_template(
            "results.html",
            summary=summary,
            rows=rows,
            charts={},
            batch_id=batch_id,
            source_type=source_type,
        )

    @app.route("/logout", methods=["GET"])
    def logout():
        session.pop("user_id", None)
        flash("Signed out successfully.", "success")
        return redirect(url_for("index"))

    with app.app_context():
        from models import alert, log, risk_score, user  # noqa: F401
        db.create_all()
        
        # --- Automatic Schema Migration ---
        try:
            from sqlalchemy import text
            # List of new columns to check and add
            new_columns = [
                ("night_login_count", "FLOAT DEFAULT 0.0"),
                ("email_activity_count", "FLOAT DEFAULT 0.0"),
                ("usb_usage_count", "FLOAT DEFAULT 0.0")
            ]
            
            for col_name, col_type in new_columns:
                try:
                    db.session.execute(text(f"ALTER TABLE logs ADD COLUMN {col_name} {col_type}"))
                    db.session.commit()
                    logging.info(f"Migration: Added column {col_name} to logs table.")
                except Exception:
                    db.session.rollback()
                    # Skip if column already exists
                    pass
            # User migration
            new_user_columns = [
                ("first_name", "VARCHAR(100)"),
                ("last_name", "VARCHAR(100)"),
                ("otp_hash", "VARCHAR(255)"),
                ("otp_expiry", "DATETIME")
            ]
            for col_name, col_type in new_user_columns:
                try:
                    db.session.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
                    db.session.commit()
                    logging.info(f"Migration: Added column {col_name} to users table.")
                except Exception:
                    db.session.rollback()
                    pass
        except Exception as migration_error:
            logging.error(f"Migration failed: {migration_error}")

        seed_default_user()

    return app


app = create_app()


def main():
    """Run the Flask development server."""
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "true").lower() == "true",
    )


if __name__ == "__main__":
    main()
