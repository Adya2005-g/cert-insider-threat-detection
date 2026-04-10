from functools import wraps
import logging
import os
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
from services.anomaly_detection import detect_anomalies as detect_pipeline_anomalies
from services.data_pipeline import feature_engineering as batch_feature_engineering
from services.data_pipeline import load_csv_dataset
from services.data_pipeline import preprocess as batch_preprocess
from services.ml_model import calculate_risk_score, classify_threat, detect_anomalies
from services.behavior_profile import create_user_profile
from services.classification import classify_threat as classify_pipeline_threat
from services.data_loader import load_data
from services.evaluate_model import evaluate
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
    return {
        "user": user,
        "login_frequency": round(float(log.login_frequency), 2),
        "after_hours_activity": round(float(log.after_hours_activity), 2),
        "file_access_count": round(float(log.file_access_count), 2),
        "threat_level": str(log.threat_level).title(),
        "risk_score": round(float(score), 2),
        "status": "Threat" if log.anomaly_flag else "Normal",
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
        "modules": _dashboard_modules(),
        "recent_alerts": [_serialize_dashboard_alert(alert) for alert in recent_alerts],
        "latest_batch_id": batch_id,
        "latest_results": latest_rows,
    }


def _process_uploaded_dataset(file_storage):
    dataset = load_csv_dataset(file_storage)
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
            file_access_count=float(row["file_access_count"]),
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

    @app.context_processor
    def inject_current_user():
        user = _current_user()
        return {"current_user": user or User.anonymous()}

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
                password_hash=generate_password_hash(password),
                role="analyst",
            )
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

        return render_template("detection.html", analysis=None)

    @app.route("/detect", methods=["POST"])
    @_login_required
    def predict():
        try:
            login_frequency = float(request.form.get("login_frequency", 0))
            after_hours_activity = float(request.form.get("after_hours_activity", 0))
            file_access_count = float(request.form.get("file_access_count", 0))
        except ValueError:
            flash("Enter numeric values for all analysis fields.", "error")
            return render_template("detection.html", analysis=None), 400

        analysis_input = pd.DataFrame(
            [
                {
                    "login_frequency": login_frequency,
                    "after_hours_activity": after_hours_activity,
                    "file_access_count": file_access_count,
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
            return render_template("results.html", summary=None, rows=[], batch_id=None)

        logs = (
            Log.query.filter_by(batch_id=batch_id)
            .order_by(Log.id.asc())
            .all()
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

        return render_template("results.html", summary=summary, rows=rows, batch_id=batch_id)

    @app.route("/logout", methods=["GET"])
    def logout():
        session.pop("user_id", None)
        flash("Signed out successfully.", "success")
        return redirect(url_for("login_page"))

    with app.app_context():
        from models import alert, log, risk_score, user  # noqa: F401

        db.create_all()
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
