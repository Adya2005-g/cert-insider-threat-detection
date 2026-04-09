from __future__ import annotations

from uuid import uuid4

import pandas as pd
from flask import Blueprint, jsonify, request
from werkzeug.security import generate_password_hash

from alerts.manager import generate_alerts
from models.alert import Alert
from models.log import Log
from models.risk_score import RiskScore
from models.user import User
from services.auth_service import authenticate_user
from services.data_pipeline import feature_engineering, load_csv_dataset, preprocess
from services.ml_model import calculate_risk_score, classify_threat, detect_anomalies
from utils.constants import ALERT_THRESHOLD
from utils.extensions import db


api_bp = Blueprint("api", __name__)


def _serialize_log(log: Log) -> dict:
    external_user_id = log.raw_record.get("user_id", log.user_id) if log.raw_record else log.user_id
    return {
        "user_id": external_user_id,
        "anomaly_flag": log.anomaly_flag,
        "threat_level": log.threat_level,
        "risk_score": log.risk_score.score if log.risk_score else 0.0,
    }


def _latest_batch_id():
    latest_log = Log.query.order_by(Log.created_at.desc(), Log.id.desc()).first()
    return latest_log.batch_id if latest_log else None


@api_bp.post("/login")
def login():
    payload = request.get_json(silent=True) or {}
    identifier = payload.get("username") or payload.get("email") or ""
    password = payload.get("password") or ""

    user = authenticate_user(identifier, password)
    if user is None:
        return jsonify({"message": "Invalid credentials."}), 401

    return jsonify({"message": "Authentication successful.", "user": user.to_dict()})


@api_bp.post("/upload")
def upload_dataset():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"message": "A CSV file is required."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"message": "Only CSV files are supported."}), 400

    dataset = load_csv_dataset(file)
    if dataset.empty:
        return jsonify({"message": "Uploaded dataset is empty."}), 400

    processed = preprocess(dataset)
    engineered = feature_engineering(processed)
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
            source_filename=file.filename,
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

    results = [_serialize_log(log) for log in logs_to_create]
    return (
        jsonify(
            {
                "message": "Dataset processed successfully.",
                "batch_id": batch_id,
                "records_processed": len(results),
                "alerts_generated": len(generated_alerts),
                "results": results,
            }
        ),
        201,
    )


@api_bp.get("/detect")
def detect():
    batch_id = request.args.get("batch_id") or _latest_batch_id()
    if batch_id is None:
        return jsonify({"message": "No processed dataset found."}), 404

    logs = (
        Log.query.filter_by(batch_id=batch_id)
        .order_by(Log.id.asc())
        .all()
    )

    results = [_serialize_log(log) for log in logs]
    return jsonify({"batch_id": batch_id, "results": results})


@api_bp.get("/risk")
def risk():
    batch_id = request.args.get("batch_id") or _latest_batch_id()
    if batch_id is None:
        return jsonify({"message": "No risk scores available."}), 404

    logs = (
        Log.query.filter_by(batch_id=batch_id)
        .order_by(Log.id.asc())
        .all()
    )
    alerts = (
        Alert.query.join(Log, Alert.log_id == Log.id)
        .filter(Log.batch_id == batch_id)
        .order_by(Alert.id.asc())
        .all()
    )

    results = [_serialize_log(log) for log in logs]
    return jsonify(
        {
            "batch_id": batch_id,
            "threshold": ALERT_THRESHOLD,
            "results": results,
            "alerts": [alert.to_dict() for alert in alerts],
        }
    )


def _json_safe_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
