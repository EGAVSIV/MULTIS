# ==========================================
# utils.py
# Security & Utility Functions
# ==========================================

import bcrypt
import streamlit as st
from datetime import datetime
from config import STATUS_APPROVED


# ==========================================
# HASH PASSWORD
# ==========================================

def hash_password(password: str) -> str:
    """
    Convert plain password to bcrypt hash
    """
    hashed = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )

    return hashed.decode("utf-8")


# ==========================================
# VERIFY PASSWORD
# ==========================================

def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify password against stored hash
    """

    try:

        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )

    except Exception:
        return False


# ==========================================
# SUBSCRIPTION VALID
# ==========================================

def is_subscription_valid(expiry_date):

    if expiry_date is None:
        return False

    try:

        expiry = datetime.strptime(
            expiry_date,
            "%Y-%m-%d"
        )

        return expiry >= datetime.now()

    except Exception:

        return False


# ==========================================
# USER APPROVED ?
# ==========================================

def is_user_approved(status):

    return status == STATUS_APPROVED


# ==========================================
# LOGIN USER
# ==========================================

def login_user(user):

    st.session_state.authenticated = True

    st.session_state.user_id = user["id"]

    st.session_state.username = user["username"]

    st.session_state.fullname = user["fullname"]

    st.session_state.role = user["role"]

    st.session_state.status = user["status"]

    st.session_state.expiry_date = user["expiry_date"]


# ==========================================
# LOGOUT
# ==========================================

def logout():

    keys = [

        "authenticated",

        "user_id",

        "username",

        "fullname",

        "role",

        "status",

        "expiry_date"

    ]

    for key in keys:

        if key in st.session_state:

            del st.session_state[key]

    st.rerun()


# ==========================================
# DAYS LEFT
# ==========================================

def days_left(expiry_date):

    try:

        expiry = datetime.strptime(
            expiry_date,
            "%Y-%m-%d"
        )

        return (expiry - datetime.now()).days

    except Exception:

        return 0


# ==========================================
# GREETING
# ==========================================

def greeting():

    hour = datetime.now().hour

    if hour < 12:
        return "🌅 Good Morning"

    elif hour < 17:
        return "☀️ Good Afternoon"

    else:
        return "🌙 Good Evening"


# ==========================================
# USER INFO BOX
# ==========================================

def user_sidebar():

    if "authenticated" not in st.session_state:
        return

    with st.sidebar:

        st.success(f"Welcome {st.session_state.fullname}")

        st.write(f"**User :** {st.session_state.username}")

        st.write(f"**Role :** {st.session_state.role}")

        st.write(
            f"**Expiry :** {st.session_state.expiry_date}"
        )

        st.write(
            f"**Days Left :** {days_left(st.session_state.expiry_date)}"
        )

        if st.button("🚪 Logout"):

            logout()
