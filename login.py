# ==========================================
# login.py
# Author : Gaurav Singh Yaadav
# ==========================================

import streamlit as st

from database import get_user

from utils import (
    verify_password,
    login_user,
    is_user_approved,
    is_subscription_valid
)

from config import (
    LOGIN_TITLE,
    LOGIN_MESSAGE
)


# ==========================================
# LOGIN PAGE
# ==========================================

def login_page():

    # --------------------------------------
    # Session Variables
    # --------------------------------------

    if "forgot_password" not in st.session_state:
        st.session_state.forgot_password = False

    if "show_register" not in st.session_state:
        st.session_state.show_register = False

    # --------------------------------------
    # Forgot Password Page
    # --------------------------------------

    if st.session_state.forgot_password:

        from forgot_password import forgot_password_page

        forgot_password_page()

        return

    # --------------------------------------
    # Register Page
    # --------------------------------------

    if st.session_state.show_register:

        from register import register_page

        register_page()

        return

    # --------------------------------------
    # Login Screen
    # --------------------------------------

    st.title(LOGIN_TITLE)

    st.markdown("### Welcome to NSE Stock Market Scanner")

    username = st.text_input(
        "👤 Username"
    )

    password = st.text_input(
        "🔑 Password",
        type="password"
    )

    st.write("")

    # --------------------------------------
    # Buttons
    # --------------------------------------

    col1, col2 = st.columns(2)

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

    st.write("")

    forgot_btn = st.button(
        "🔑 Forgot Password ?",
        use_container_width=True
    )

    # --------------------------------------
    # Forgot Password
    # --------------------------------------

    if forgot_btn:

        st.session_state.forgot_password = True

        st.rerun()

    # --------------------------------------
    # Register
    # --------------------------------------

    if register_btn:

        st.session_state.show_register = True

        st.rerun()

    # --------------------------------------
    # LOGIN
    # --------------------------------------

    if login_btn:

        username = username.strip()

        password = password.strip()

        if username == "" or password == "":

            st.error(
                "Please enter Username and Password."
            )

            return

        user = get_user(username)

        if user is None:

            st.error(
                "Username not found."
            )

            return

        if not verify_password(
            password,
            user["password"]
        ):

            st.error(
                "Incorrect Password."
            )

            return

        if not is_user_approved(
            user["status"]
        ):

            st.warning(
                "Your account is waiting for Admin Approval."
            )

            return

        if not is_subscription_valid(
            user["expiry_date"]
        ):

            st.error(
                "Your subscription has expired."
            )

            return

        login_user(user)

        st.success(
            f"Welcome {user['fullname']}"
        )

        st.rerun()

    # --------------------------------------
    # LOGIN MESSAGE
    # --------------------------------------

    st.divider()

    st.info(LOGIN_MESSAGE)

    st.caption(
        "© 2026 Gaurav Singh Yaadav | NSE Stock Market Scanner"
    )
