import sys
import os
# Add the project root to sys.path
sys.path.append(os.getcwd())

from flask import Flask
from utils.extensions import db
from models.user import User
from utils.config import DevelopmentConfig

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
db.init_app(app)

with app.app_context():
    # Test User
    u = User(username="testuser", email="test@example.com")
    
    # Test Password Hashing
    print("Testing Password Hashing...")
    u.set_password("mypassword")
    assert u.password_hash.startswith("$2") # Bcrypt header
    assert u.check_password("mypassword") is True
    assert u.check_password("wrong") is False
    print("Password Hashing [OK]")
    
    # Test OTP Hashing
    print("Testing OTP Hashing...")
    u.set_otp("123456")
    assert u.otp_hash.startswith("$2")
    assert u.verify_otp("123456") is True
    assert u.verify_otp("000000") is False
    print("OTP Hashing [OK]")
    
    # Test OTP Expiry
    print("Testing OTP Expiry...")
    from datetime import datetime, timedelta
    u.otp_expiry = datetime.utcnow() - timedelta(minutes=1)
    assert u.verify_otp("123456") is False
    print("OTP Expiry [OK]")

print("All Sanity Checks Passed!")
