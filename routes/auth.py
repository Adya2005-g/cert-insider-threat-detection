import secrets
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for, session
from models.user import User
from utils.extensions import db
from services.mail_service import send_otp_email

auth_bp = Blueprint("auth_new", __name__)

@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Step 1: Request OTP."""
    email = request.form.get("email", "").strip().lower()
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"error": "No account found with this email."}), 404

    # Generate 6-digit OTP
    otp = f"{secrets.randbelow(1000000):06d}"
    user.set_otp(otp)
    db.session.commit()

    if send_otp_email(email, otp):
        return jsonify({"message": "Verification code sent to your email."}), 200
    else:
        # Fallback for demo if SMTP fails
        return jsonify({
            "message": "OTP sent (Demo Fallback)",
            "demo_otp": otp,
            "warning": "Email delivery failed. Use demo OTP."
        }), 200

@auth_bp.route("/verify-otp", methods=["POST"])
def verify_otp():
    """Step 2: Verify OTP."""
    email = request.form.get("email", "").strip().lower()
    otp = request.form.get("otp", "").strip()
    
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found."}), 404

    if user.verify_otp(otp):
        # Store in session that this email is "verified" for reset
        session["verified_email"] = email
        session["otp_verified_at"] = datetime.utcnow().isoformat()
        return jsonify({"message": "OTP verified successfully."}), 200
    else:
        return jsonify({"error": "Invalid or expired verification code."}), 400

@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    """Step 3: Reset Password."""
    email = request.form.get("email", "").strip().lower()
    new_password = request.form.get("password", "")
    
    # Check if this email was recently verified
    if session.get("verified_email") != email:
         return jsonify({"error": "Unauthorized. Please verify OTP first."}), 401

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found."}), 404

    user.set_password(new_password)
    # Clear OTP columns
    user.otp_hash = None
    user.otp_expiry = None
    db.session.commit()
    
    session.pop("verified_email", None)
    session.pop("otp_verified_at", None)
    
    return jsonify({"message": "Password reset successful. You can now login."}), 200

@auth_bp.route("/resend-otp", methods=["POST"])
def resend_otp():
    """Bonus: Resend OTP API."""
    email = request.form.get("email", "").strip().lower()
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"error": "User not found."}), 404

    # Check cooldown (e.g. 30 seconds)
    # For simplicity, we just resend if requested
    otp = f"{secrets.randbelow(1000000):06d}"
    user.set_otp(otp)
    db.session.commit()

    if send_otp_email(email, otp):
        return jsonify({"message": "New verification code sent."}), 200
    return jsonify({"error": "Failed to send email."}), 500
