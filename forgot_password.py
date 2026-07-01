# ==========================================
# forgot_password.py
# Part-1
# Author : Gaurav Singh Yaadav
# ==========================================

import streamlit as st

from database import (
    get_user_by_username_email,
    save_otp
)

from otp_utils import (
    generate_otp,
    otp_expiry_time
)

from email_service import send_otp_email


# ==========================================
# FORGOT PASSWORD PAGE
# ==========================================

def forgot_password_page():

    st.title("🔐 Forgot Password")

    st.info(
        "Enter your Username and Registered Email ID."
    )

    # ---------------------------------
    # Session
    # ---------------------------------

    if "otp_sent" not in st.session_state:

        st.session_state.otp_sent = False

    if "reset_username" not in st.session_state:

        st.session_state.reset_username = ""

    if "reset_email" not in st.session_state:

        st.session_state.reset_email = ""

    # ---------------------------------
    # STEP-1
    # ---------------------------------

    if not st.session_state.otp_sent:

        username = st.text_input(
            "👤 Username"
        ).strip()

        email = st.text_input(
            "📧 Registered Email"
        ).strip().lower()

        if st.button(
            "📨 Send OTP",
            use_container_width=True
        ):

            if username == "" or email == "":

                st.warning(
                    "Please enter Username and Email."
                )

                st.stop()

            user = get_user_by_username_email(
                username,
                email
            )

            if user is None:

                st.error(
                    "Username or Email not found."
                )

                st.stop()

            otp = generate_otp()

            expiry = otp_expiry_time()

            save_otp(
                username,
                otp,
                expiry
            )

            ok = send_otp_email(

                recipient=email,

                fullname=user["fullname"],

                otp=otp

            )

            if ok:

                st.session_state.otp_sent = True

                st.session_state.reset_username = username

                st.session_state.reset_email = email

                st.success(
                    "OTP sent successfully."
                )

                st.rerun()

            else:

                st.error(
                    "Unable to send OTP."
                )

        return


    # ==========================================
    # STEP-2
    # VERIFY OTP
    # ==========================================

    from database import (
        verify_saved_otp,
        clear_otp,
        update_password
    )

    from otp_utils import (
        is_otp_valid
    )

    from utils import hash_password

    from email_service import (
        send_password_changed_email
    )

    st.success(
        f"OTP has been sent to\n\n{st.session_state.reset_email}"
    )

    st.info(
        "Please enter the OTP received on your email."
    )

    otp = st.text_input(

        "🔐 Enter OTP",

        max_chars=6

    )

    st.divider()

    password = st.text_input(

        "🔑 New Password",

        type="password"

    )

    confirm = st.text_input(

        "🔑 Confirm Password",

        type="password"

    )

    col1, col2 = st.columns(2)

    # ---------------------------------
    # VERIFY OTP
    # ---------------------------------

    with col1:

        if st.button(

            "✅ Verify & Reset Password",

            use_container_width=True

        ):

            if otp == "":

                st.warning(
                    "Enter OTP."
                )

                st.stop()

            ok, result = verify_saved_otp(

                st.session_state.reset_username,

                otp

            )

            if not ok:

                st.error(result)

                st.stop()

            if not is_otp_valid(result):

                st.error(

                    "OTP has expired."

                )

                clear_otp(

                    st.session_state.reset_username

                )

                st.stop()

            if len(password) < 8:

                st.warning(

                    "Password must contain at least 8 characters."

                )

                st.stop()

            if password != confirm:

                st.error(

                    "Passwords do not match."

                )

                st.stop()

            hashed = hash_password(

                password

            )

            update_password(

                st.session_state.reset_username,

                hashed

            )

            clear_otp(

                st.session_state.reset_username

            )

            user = get_user_by_username_email(

                st.session_state.reset_username,

                st.session_state.reset_email

            )

            send_password_changed_email(

                st.session_state.reset_email,

                user["fullname"]

            )

            st.session_state.password_reset = True

            st.rerun()

    # ---------------------------------
    # RESEND OTP
    # ---------------------------------

    with col2:

        if st.button(

            "🔄 Resend OTP",

            use_container_width=True

        ):

            user = get_user_by_username_email(

                st.session_state.reset_username,

                st.session_state.reset_email

            )

            otp = generate_otp()

            expiry = otp_expiry_time()

            save_otp(

                st.session_state.reset_username,

                otp,

                expiry

            )

            send_otp_email(

                recipient=st.session_state.reset_email,

                fullname=user["fullname"],

                otp=otp

            )

            st.success(

                "New OTP sent successfully."

            )

            st.rerun()


    # ==========================================
    # PASSWORD RESET SUCCESS
    # ==========================================

    if "password_reset" not in st.session_state:

        st.session_state.password_reset = False

    if st.session_state.password_reset:

        st.balloons()

        st.success(
            "🎉 Password Changed Successfully!"
        )

        st.info(
            f"""
Username :

{st.session_state.reset_username}

You can now login using your new password.
"""
        )

        col1, col2 = st.columns(2)

        # ---------------------------------
        # BACK TO LOGIN
        # ---------------------------------

        with col1:

            if st.button(
                "🔐 Back to Login",
                use_container_width=True
            ):

                keys = [

                    "otp_sent",

                    "password_reset",

                    "reset_username",

                    "reset_email"

                ]

                for key in keys:

                    if key in st.session_state:

                        del st.session_state[key]

                st.session_state.show_login = True

                st.rerun()

        # ---------------------------------
        # RESET AGAIN
        # ---------------------------------

        with col2:

            if st.button(
                "🔄 Reset Another Password",
                use_container_width=True
            ):

                keys = [

                    "otp_sent",

                    "password_reset",

                    "reset_username",

                    "reset_email"

                ]

                for key in keys:

                    if key in st.session_state:

                        del st.session_state[key]

                st.rerun()
