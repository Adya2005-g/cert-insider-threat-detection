from sqlalchemy.sql import func

from utils.extensions import db


class Log(db.Model):
    __tablename__ = "logs"

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(64), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    event_timestamp = db.Column(db.DateTime, nullable=True, index=True)
    login_frequency = db.Column(db.Float, nullable=False, default=0.0)
    after_hours_activity = db.Column(db.Float, nullable=False, default=0.0)
    night_login_count = db.Column(db.Float, nullable=False, default=0.0)
    file_access_count = db.Column(db.Float, nullable=False, default=0.0)
    email_activity_count = db.Column(db.Float, nullable=False, default=0.0)
    usb_usage_count = db.Column(db.Float, nullable=False, default=0.0)
    anomaly_flag = db.Column(db.Boolean, nullable=False, default=False)
    threat_level = db.Column(db.String(20), nullable=False, default="low")
    source_filename = db.Column(db.String(255), nullable=True)
    raw_record = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())

    user = db.relationship("User", back_populates="logs")
    risk_score = db.relationship("RiskScore", back_populates="log", uselist=False, cascade="all, delete-orphan")
    alerts = db.relationship("Alert", back_populates="log", lazy=True, cascade="all, delete-orphan")

    def to_result_dict(self):
        return {
            "log_id": self.id,
            "user_id": self.user_id,
            "anomaly_flag": self.anomaly_flag,
            "threat_level": self.threat_level,
            "risk_score": self.risk_score.score if self.risk_score else None,
            "batch_id": self.batch_id,
        }
