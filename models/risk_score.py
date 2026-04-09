from sqlalchemy.sql import func

from utils.extensions import db


class RiskScore(db.Model):
    __tablename__ = "risk_scores"

    id = db.Column(db.Integer, primary_key=True)
    log_id = db.Column(db.Integer, db.ForeignKey("logs.id"), nullable=False, unique=True, index=True)
    score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())

    log = db.relationship("Log", back_populates="risk_score")
