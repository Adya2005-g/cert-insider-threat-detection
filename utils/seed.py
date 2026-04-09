from werkzeug.security import generate_password_hash

from models.user import User
from utils.extensions import db


def seed_default_user():
    """Ensure the API always has one working analyst account."""
    existing = User.query.filter_by(username="admin").first()
    if existing:
        return existing

    user = User(
        username="admin",
        email="admin@example.com",
        password_hash=generate_password_hash("admin123"),
        role="admin",
    )
    db.session.add(user)
    db.session.commit()
    return user
