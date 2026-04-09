from werkzeug.security import check_password_hash

from models.user import User


def authenticate_user(username_or_email: str, password: str):
    """Authenticate a user using username or email."""
    identifier = (username_or_email or "").strip().lower()
    if not identifier or not password:
        return None

    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()

    if user and check_password_hash(user.password_hash, password):
        return user
    return None
