from sqlalchemy.sql import func

from utils.extensions import db


class Alert(db.Model):
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    log_id = db.Column(db.Integer, db.ForeignKey("logs.id"), nullable=False, index=True)
    risk_score = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    message = db.Column(db.String(255), nullable=False)
    severity = db.Column(db.String(20), nullable=False, default="high")
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())

    log = db.relationship("Log", back_populates="alerts")

    def to_dict(self):
        return {
            "id": self.id,
            "log_id": self.log_id,
            "risk_score": self.risk_score,
            "threshold": self.threshold,
            "message": self.message,
            "severity": self.severity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
