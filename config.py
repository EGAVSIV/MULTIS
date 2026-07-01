# ==========================================
# config.py
# Configuration File
# Author : Gaurav Singh Yaadav
# ==========================================

import os
import streamlit as st

# ------------------------------------------
# APP INFORMATION
# ------------------------------------------

APP_NAME = "NSE Stock Market Scanner"
APP_VERSION = "1.0.0"

# ------------------------------------------
# DATABASE
# ------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_NAME = "users.db"

DATABASE_PATH = os.path.join(BASE_DIR, DATABASE_NAME)

# ------------------------------------------
# ADMIN DETAILS
# ------------------------------------------

ADMIN_USERNAME = "admin"

ADMIN_ROLE = "Admin"

USER_ROLE = "User"

# ------------------------------------------
# USER STATUS
# ------------------------------------------

STATUS_PENDING = "Pending"

STATUS_APPROVED = "Approved"

STATUS_REJECTED = "Rejected"

STATUS_DISABLED = "Disabled"

# ------------------------------------------
# DEFAULT SUBSCRIPTION
# ------------------------------------------

DEFAULT_SUBSCRIPTION_DAYS = 30

# ------------------------------------------
# PASSWORD SETTINGS
# ------------------------------------------

PASSWORD_MIN_LENGTH = 8

# ------------------------------------------
# SESSION SETTINGS
# ------------------------------------------

SESSION_TIMEOUT_MINUTES = 60

# ------------------------------------------
# CONTACT INFORMATION
# ------------------------------------------

OWNER_NAME = "Gaurav Singh Yaadav"

OWNER_PHONE = "+91 80039 94518"

OWNER_EMAIL = "yourmail@gmail.com"

# ------------------------------------------
# UI SETTINGS
# ------------------------------------------

APP_ICON = "📈"

LOGIN_TITLE = "🔐 Secure Login"

REGISTER_TITLE = "📝 Create Account"

ADMIN_TITLE = "👨‍💼 Admin Dashboard"

# ------------------------------------------
# LOGIN MESSAGE
# ------------------------------------------

LOGIN_MESSAGE = f"""
### Need Login Access?

If you don't have Username and Password,

📞 Contact **{OWNER_NAME}**

📱 Mobile / WhatsApp

**{OWNER_PHONE}**
"""

# ------------------------------------------
# ROLES
# ------------------------------------------

ROLES = [
    ADMIN_ROLE,
    USER_ROLE
]

# ------------------------------------------
# STATUS LIST
# ------------------------------------------

STATUS_LIST = [
    STATUS_PENDING,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_DISABLED
]

# ----------------------------
# EMAIL SUBJECTS
# ----------------------------

EMAIL_SUBJECT_WELCOME = "Welcome to NSE Stock Market Scanner"

EMAIL_SUBJECT_OTP = "Password Reset OTP"

EMAIL_SUBJECT_RESET = "Password Changed Successfully"

# ----------------------------
# APPLICATION URL
# ----------------------------

APP_URL = "https://multis.streamlit.app/"


# ----------------------------
# EMAIL CONFIGURATION
# ----------------------------



EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ----------------------------
# OTP SETTINGS
# ----------------------------

OTP_LENGTH = 6
OTP_VALIDITY_MINUTES = 10
OTP_MAX_ATTEMPTS = 5

