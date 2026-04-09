from functools import wraps
import os

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import generate_password_hash

from models.user import User
from routes.api import api_bp
from services.auth_service import authenticate_user
from utils.config import DevelopmentConfig
from utils.extensions import db
from utils.seed import seed_default_user


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
        return render_template("dashboard.html")

    @app.route("/detection", methods=["GET"])
    @app.route("/detect", methods=["GET"])
    @_login_required
    def detect_view():
        return render_template("detection.html")

    @app.route("/upload", methods=["GET"])
    @_login_required
    def upload():
        return render_template("upload.html")

    @app.route("/results", methods=["GET"])
    @_login_required
    def results():
        return render_template("results.html", tables=None, summary=None)

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
