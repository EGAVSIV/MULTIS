# ==========================================
# email_service.py
# Author : Gaurav Singh Yaadav
# ==========================================

import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import (
    EMAIL_ADDRESS,
    EMAIL_PASSWORD,
    SMTP_SERVER,
    SMTP_PORT,
    APP_NAME,
    OWNER_NAME,
    APP_URL
)


# ==========================================
# SEND EMAIL
# ==========================================

def send_email(recipient, subject, body):

    try:

        message = MIMEMultipart()

        message["From"] = EMAIL_ADDRESS
        message["To"] = recipient
        message["Subject"] = subject

        message.attach(
            MIMEText(body, "plain")
        )

        server = smtplib.SMTP(
            SMTP_SERVER,
            SMTP_PORT
        )

        server.starttls()

        server.login(
            EMAIL_ADDRESS,
            EMAIL_PASSWORD
        )

        server.sendmail(
            EMAIL_ADDRESS,
            recipient,
            message.as_string()
        )

        server.quit()

        return True

    except Exception as e:

        import traceback

        st.error(f"Email Error: {e}")

        st.code(traceback.format_exc())

        return False


# ==========================================
# SEND OTP EMAIL
# ==========================================

def send_otp_email(
        recipient,
        fullname,
        otp
):

    subject = f"{APP_NAME} - Password Reset OTP"

    body = f"""
Hello {fullname},

We received a request to reset your password.

Your OTP is

========================

{otp}

========================

This OTP is valid for 10 minutes.

If you did not request this password reset,
please ignore this email.

Login URL

{APP_URL}

Regards,

{OWNER_NAME}
"""

    return send_email(
        recipient,
        subject,
        body
    )


# ==========================================
# SEND WELCOME EMAIL
# ==========================================

def send_welcome_email(
        recipient,
        fullname,
        username,
        password
):

    subject = f"Welcome to {APP_NAME}"

    body = f"""
Hello {fullname},

Welcome to {APP_NAME}.

Your account has been activated successfully.

Login Details

--------------------------------

Username

{username}

Password

{password}

--------------------------------

Login Here

{APP_URL}

Please change your password
after your first login.

Regards,

{OWNER_NAME}
"""

    return send_email(
        recipient,
        subject,
        body
    )


# ==========================================
# PASSWORD RESET SUCCESS
# ==========================================

def send_password_changed_email(
        recipient,
        fullname
):

    subject = "Password Changed Successfully"

    body = f"""
Hello {fullname},

Your password has been changed successfully.

If this wasn't you,
please contact the administrator immediately.

Login

{APP_URL}

Regards,

{OWNER_NAME}
"""

    return send_email(
        recipient,
        subject,
        body
    )


# ==========================================
# ACCOUNT APPROVED
# ==========================================

def send_account_approved_email(
        recipient,
        fullname
):

    subject = "Account Approved"

    body = f"""
Hello {fullname},

Congratulations!

Your account has been approved.

You can now login using your credentials.

Login

{APP_URL}

Regards,

{OWNER_NAME}
"""

    return send_email(
        recipient,
        subject,
        body
    )
