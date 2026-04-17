from datetime import datetime
import secrets

from sqlalchemy.sql import func
from werkzeug.security import check_password_hash, generate_password_hash

from utils.extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    role = db.Column(db.String(50), nullable=False, default="analyst")
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())
    
    otp_hash = db.Column(db.String(255), nullable=True)
    otp_expiry = db.Column(db.DateTime, nullable=True)

    logs = db.relationship("Log", back_populates="user", lazy=True)

    def set_password(self, password):
        """Hash password using Werkzeug's built-in secure hash."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check password without requiring bcrypt to be installed."""
        try:
            return check_password_hash(self.password_hash, password)
        except Exception:
            # Support environments that may already contain bcrypt-style hashes.
            try:
                import bcrypt

                return bcrypt.checkpw(password.encode("utf-8"), self.password_hash.encode("utf-8"))
            except Exception:
                return False

    def set_otp(self, otp):
        """Hash and store OTP with expiry."""
        from datetime import datetime, timedelta
        self.otp_hash = generate_password_hash(otp)
        self.otp_expiry = datetime.utcnow() + timedelta(minutes=5)

    def verify_otp(self, otp):
        """Verify OTP hash and check expiry."""
        if not self.otp_hash or not self.otp_expiry:
            return False
        if datetime.utcnow() > self.otp_expiry:
            return False
        try:
            return check_password_hash(self.otp_hash, otp)
        except Exception:
            return secrets.compare_digest(self.otp_hash, otp)

    @property
    def fname(self):
        if self.first_name:
            return self.first_name.capitalize()
        parts = self.username.replace("_", ".").split(".")
        return parts[0].capitalize() if parts and parts[0] else "User"

    @property
    def lname(self):
        if self.last_name:
            return self.last_name.capitalize()
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
