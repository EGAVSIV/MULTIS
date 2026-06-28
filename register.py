# ==========================================
# register.py
# User Registration
# ==========================================

import streamlit as st

from database import register_user
from utils import hash_password

from config import (
    REGISTER_TITLE,
    PASSWORD_MIN_LENGTH,
    OWNER_NAME,
    OWNER_PHONE,
)

# ==========================================
# REGISTER PAGE
# ==========================================

def register_page():

    st.title(REGISTER_TITLE)

    st.write("Create your account to request access.")

    with st.form("register_form", clear_on_submit=True):

        fullname = st.text_input("Full Name")

        mobile = st.text_input("Mobile Number")

        email = st.text_input("Email Address")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "📝 Register"
        )

    if submitted:

        # -------------------------
        # Validation
        # -------------------------

        if fullname.strip() == "":
            st.error("Please enter Full Name.")
            return

        if username.strip() == "":
            st.error("Please enter Username.")
            return

        if mobile.strip() == "":
            st.error("Please enter Mobile Number.")
            return

        if len(password) < PASSWORD_MIN_LENGTH:
            st.error(
                f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
            )
            return

        if password != confirm_password:
            st.error("Passwords do not match.")
            return

        # -------------------------
        # Hash Password
        # -------------------------

        hashed_password = hash_password(password)

        # -------------------------
        # Save User
        # -------------------------

        success = register_user(
            username=username,
            password=hashed_password,
            fullname=fullname,
            mobile=mobile,
            email=email
        )

        if success:

            st.success(
                "✅ Registration Successful."
            )

            st.info(
                """
Your account has been submitted for Admin Approval.

You will be able to login once your account has been approved.
"""
            )

            st.markdown("---")

            st.write("### Contact")

            st.write(f"**{OWNER_NAME}**")

            st.write(f"📞 {OWNER_PHONE}")

        else:

            st.error(
                "Username already exists."
            )

    st.markdown("---")

    if st.button("⬅ Back to Login"):

        st.session_state.show_register = False

        st.rerun()
