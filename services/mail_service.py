import logging
import os
import smtplib
from email.message import EmailMessage

def send_otp_email(recipient, otp):
    """
    Send a 6-digit OTP to the recipient's email using SMTP.
    Config is pulled from environment variables.
    """
    smtp_host = os.environ.get("MAIL_SERVER")
    smtp_port = int(os.environ.get("MAIL_PORT", "587"))
    smtp_username = os.environ.get("MAIL_USERNAME")
    smtp_password = os.environ.get("MAIL_PASSWORD")
    sender = os.environ.get("MAIL_DEFAULT_SENDER", smtp_username)
    use_tls = os.environ.get("MAIL_USE_TLS", "true").lower() == "true"

    if not all([smtp_host, smtp_username, smtp_password, sender]):
        logging.warning("SMTP configuration is incomplete. OTP: %s", otp)
        return False

    msg = EmailMessage()
    msg["Subject"] = "Your CERT Insider Verification Code"
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(
        f"Hello,\n\n"
        f"Your verification code is: {otp}\n\n"
        f"This code is valid for 5 minutes. If you did not request this, please ignore this email.\n\n"
        f"Stay secure,\n"
        f"CERT Insider Security Team"
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            logging.info("OTP email sent successfully to %s", recipient)
            return True
    except Exception as e:
        logging.error("Failed to send OTP email: %s", e)
        return False
