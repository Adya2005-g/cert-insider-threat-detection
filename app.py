from flask import Flask, jsonify
import os

from routes.api import api_bp
from utils.config import DevelopmentConfig
from utils.extensions import db
from utils.seed import seed_default_user


def create_app(config_object=DevelopmentConfig):
    """Application factory used by Flask and tests."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    db.init_app(app)
    app.register_blueprint(api_bp)

    @app.route("/", methods=["GET"])
    def healthcheck():
        return jsonify(
            {
                "service": "Insider Threat Detection System",
                "status": "ok",
                "available_endpoints": ["/login", "/upload", "/detect", "/risk"],
            }
        )

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
