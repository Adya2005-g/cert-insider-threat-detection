import os


class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///insider_threat.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_SORT_KEYS = False


class DevelopmentConfig(BaseConfig):
    DEBUG = True
