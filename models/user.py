from datetime import datetime

from sqlalchemy.sql import func

from utils.extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="analyst")
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())

    logs = db.relationship("Log", back_populates="user", lazy=True)

    @property
    def fname(self):
        parts = self.username.replace("_", ".").split(".")
        return parts[0].capitalize() if parts and parts[0] else "User"

    @property
    def lname(self):
        parts = self.username.replace("_", ".").split(".")
        return parts[1].capitalize() if len(parts) > 1 and parts[1] else ""

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else None,
        }

    @staticmethod
    def anonymous():
        class AnonymousUser:
            is_authenticated = False
            fname = ""
            lname = ""
            username = "guest"

        return AnonymousUser()
