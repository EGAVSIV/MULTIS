# ==========================================
# login.py
# ==========================================

import streamlit as st
from forgot_password import forgot_password_page

from database import get_user
from utils import (
    verify_password,
    login_user,
    is_user_approved,
    is_subscription_valid,
)
from config import LOGIN_TITLE, LOGIN_MESSAGE


def login_page():
    # ----------------------------------------
# Forgot Password Session
# ----------------------------------------

    if "forgot_password" not in st.session_state:

        st.session_state.forgot_password = False

    if st.session_state.forgot_password:

        from forgot_password import forgot_password_page

        forgot_password_page()

        return

    # -----------------------------
    # Register Page
    # -----------------------------
    if st.session_state.get("show_register", False):
        from register import register_page      # Import here to avoid circular import
        register_page()
        return

    st.title(LOGIN_TITLE)

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        login_btn = st.button(
            "🔐 Login",
            use_container_width=True
        )

    with col2:
        register_btn = st.button(
            "📝 Create Account",
            use_container_width=True
        )

    with col3:
        forgot_btn = st.button(
            "🔑 Forgot Password",
            use_container_width=True
        )

     if forgot_btn:

        st.session_state.forgot_password = True

        st.rerun()

        
    # -----------------------------
    # Open Register Page
    # -----------------------------
    if register_btn:

        st.session_state.show_register = True
        st.rerun()

    # -----------------------------
    # Login
    # -----------------------------
    if login_btn:

        if username.strip() == "" or password.strip() == "":

            st.error("Please enter Username and Password.")
            return

        user = get_user(username)

        if user is None:

            st.error("Username not found.")
            return

        if not verify_password(password, user["password"]):

            st.error("Incorrect Password.")
            return

        if not is_user_approved(user["status"]):

            st.warning("Your account is waiting for Admin approval.")
            return

        if not is_subscription_valid(user["expiry_date"]):

            st.error("Your subscription has expired.")
            return

        login_user(user)

        st.success("Login Successful")

        st.rerun()

    st.markdown("---")

    st.info(LOGIN_MESSAGE)
