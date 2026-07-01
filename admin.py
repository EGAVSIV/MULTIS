# ==========================================
# admin.py (PART-1)
# ==========================================

import streamlit as st
import pandas as pd
from database import get_user, update_password
from utils import hash_password, verify_password
from database import create_user

from database import (
    get_all_users,
    approve_user,
    delete_user,
    disable_user,
    enable_user,
    extend_subscription,
)

from utils import logout


# ==========================================
# ADMIN PANEL
# ==========================================

def admin_panel():

    # ---------------------------
    # Sidebar
    # ---------------------------

    st.sidebar.title("👨‍💼 Admin Panel")

    if st.sidebar.button("🚪 Logout"):

        logout()

    st.title("👨‍💼 NSE Scanner Admin Dashboard")

    users = get_all_users()

    # ---------------------------
    # Dashboard
    # ---------------------------

    total_users = len(users)

    pending = len(
        [u for u in users if u["status"] == "Pending"]
    )

    approved = len(
        [u for u in users if u["status"] == "Approved"]
    )

    disabled = len(
        [u for u in users if u["status"] == "Disabled"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Total", total_users)
    col2.metric("🟡 Pending", pending)
    col3.metric("🟢 Approved", approved)
    col4.metric("🔴 Disabled", disabled)

    st.divider()

    # ---------------------------
    # Tabs
    # ---------------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "👤 Create User",
            "🟡 Pending Users",
            "🟢 Approved Users",
            "🔴 Disabled Users",
            "📊 Reports"
        ]
    )

    # =====================================================
    # CREATE USER
    # =====================================================

    with tab1:

        st.subheader("👤 Create New User")

        fullname = st.text_input("Full Name")

        mobile = st.text_input("Mobile Number")

        email = st.text_input("Email")

        username = st.text_input("Username")

        password = st.text_input("Password", type="password")

        days = st.selectbox(
            "Subscription",
            [30, 90, 180, 365]
        )

        if st.button("✅ Create User"):

    # -------------------------
    # Validation
    # -------------------------

            if fullname.strip() == "":
                st.error("❌ Please enter Full Name.")

            elif username.strip() == "":
                st.error("❌ Please enter Username.")

            elif len(username) < 4:
                st.error("❌ Username must contain at least 4 characters.")

            elif " " in username:
                st.error("❌ Username cannot contain spaces.")

            elif mobile.strip() == "":
                st.error("❌ Please enter Mobile Number.")

            elif not mobile.isdigit():
                st.error("❌ Mobile Number must contain only digits.")

            elif len(mobile) != 10:
                st.error("❌ Mobile Number must be exactly 10 digits.")

            elif email.strip() == "":
                st.error("❌ Please enter Email Address.")

            elif "@" not in email:
                st.error("❌ Invalid Email Address.")

            elif password.strip() == "":
                st.error("❌ Please enter Password.")

            elif len(password) < 8:
                st.error("❌ Password must be at least 8 characters.")

            else:

                hashed = hash_password(password)

                success, message = create_user(
                    username,
                    hashed,
                    fullname,
                    mobile,
                    email,
                    days
                )

                if success:

                    st.success(message)

                    st.rerun()

                else:

                    st.error(message)
                
    # =====================================================
    # Pending Users
    # =====================================================

    with tab2:

        pending_users = [

            u for u in users

            if u["status"] == "Pending"

        ]

        if len(pending_users) == 0:

            st.success("No Pending Users.")

        else:

            for user in pending_users:

                with st.expander(

                    f"{user['fullname']} ({user['username']})"

                ):

                    st.write(f"**Name :** {user['fullname']}")
                    st.write(f"**Username :** {user['username']}")
                    st.write(f"**Mobile :** {user['mobile']}")
                    st.write(f"**Email :** {user['email']}")
                    st.write(f"**Registered :** {user['created_on']}")

                    c1, c2 = st.columns(2)

                    with c1:

                        if st.button(

                            "✅ Approve",

                            key=f"approve_{user['id']}"

                        ):

                            approve_user(user["id"])

                            # Email will be added in Part-3

                            st.success("User Approved")

                            st.rerun()

                    with c2:

                        if st.button(

                            "🗑 Delete",

                            key=f"delete_pending_{user['id']}"

                        ):

                            delete_user(user["id"])

                            st.warning("User Deleted")

                            st.rerun()

    # =====================================================
    # Approved Users
    # =====================================================

    with tab3:

        approved_users = [

            u for u in users

            if u["status"] == "Approved"

        ]

        if len(approved_users) == 0:

            st.info("No Approved Users")

        else:

            for user in approved_users:

                if user["role"] == "Admin":

                    continue

                with st.expander(

                    f"{user['fullname']} ({user['username']})"

                ):

                    st.write(f"**Mobile :** {user['mobile']}")
                    st.write(f"**Email :** {user['email']}")
                    st.write(f"**Expiry :** {user['expiry_date']}")

                    c1, c2, c3 = st.columns(3)

                    # Disable

                    with c1:

                        if st.button(

                            "🚫 Disable",

                            key=f"disable_{user['id']}"

                        ):

                            disable_user(user["id"])

                            st.success("User Disabled")

                            st.rerun()

                    # Extend Subscription

                    with c2:

                        if st.button(

                            "📅 +30 Days",

                            key=f"extend_{user['id']}"

                        ):

                            extend_subscription(

                                user["id"],

                                30

                            )

                            st.success(

                                "Subscription Extended"

                            )

                            st.rerun()

                    # Delete

                    with c3:

                        if st.button(

                            "🗑 Delete",

                            key=f"delete_{user['id']}"

                        ):

                            delete_user(

                                user["id"]

                            )

                            st.error("User Deleted")

                            st.rerun()


    # =====================================================
    # Disabled Users
    # =====================================================

    with tab4:

        disabled_users = [

            u for u in users

            if u["status"] == "Disabled"

        ]

        if len(disabled_users) == 0:

            st.success("No Disabled Users")

        else:

            for user in disabled_users:

                with st.expander(

                    f"{user['fullname']} ({user['username']})"

                ):

                    st.write(f"**Mobile :** {user['mobile']}")
                    st.write(f"**Email :** {user['email']}")

                    c1, c2 = st.columns(2)

                    # Enable

                    with c1:

                        if st.button(

                            "🟢 Enable",

                            key=f"enable_{user['id']}"

                        ):

                            enable_user(

                                user["id"]

                            )

                            st.success(

                                "User Enabled"

                            )

                            st.rerun()

                    # Delete

                    with c2:

                        if st.button(

                            "🗑 Delete",

                            key=f"delete_disabled_{user['id']}"

                        ):

                            delete_user(

                                user["id"]

                            )

                            st.warning(

                                "User Deleted"

                            )

                            st.rerun()


    # =====================================================
    # REPORTS
    # =====================================================

    with tab5:

        st.subheader("📊 User Report")

        df = pd.DataFrame([dict(u) for u in users])

        if len(df):

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # --------------------------
            # Search User
            # --------------------------

            search = st.text_input(
                "🔍 Search User"
            )

            if search:

                result = df[
                    df.astype(str)
                    .apply(
                        lambda x: x.str.contains(
                            search,
                            case=False
                        )
                    )
                    .any(axis=1)
                ]

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            # --------------------------
            # Download CSV
            # --------------------------

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                "📥 Download Users CSV",

                csv,

                file_name="users.csv",

                mime="text/csv"

            )

        else:

            st.info("No Users Found")

    st.divider()



    # =====================================================
    # ADMIN SETTINGS
    # =====================================================

    st.subheader("⚙️ Admin Settings")

    with st.expander("🔑 Change Admin Password", expanded=False):

        current_password = st.text_input(
            "Current Password",
            type="password",
            key="admin_current_pwd"
        )

        new_password = st.text_input(
            "New Password",
            type="password",
            key="admin_new_pwd"
        )

        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
            key="admin_confirm_pwd"
        )

        if st.button(
            "✅ Update Password",
            key="update_admin_password"
        ):

            admin = get_user(st.session_state.username)

            if admin is None:

                st.error("Admin user not found.")

            elif not verify_password(
                current_password,
                admin["password"]
            ):

                st.error("❌ Current password is incorrect.")

            elif new_password != confirm_password:

                st.error("❌ New passwords do not match.")

            elif len(new_password) < 8:

                st.warning(
                    "Password must be at least 8 characters."
                )

            else:

                hashed_password = hash_password(new_password)

                update_password(
                    st.session_state.username,
                    hashed_password
                )

                st.success(
                    "✅ Password updated successfully."
                )

                st.info(
                    "Please logout and login again using your new password."
                )

        st.divider()

    # ==========================================
    # EMAIL TEST
    # ==========================================

    st.subheader("📧 Email Test")

    from email_service import send_email

    test_email = st.text_input(
        "Recipient Email",
        value=st.session_state.get("username", "")
    )

    if st.button("📨 Send Test Email"):

        ok = send_email(
            test_email,
            "NSE Scanner Email Test",
            "Congratulations!\n\nYour Gmail SMTP configuration is working successfully.\n\nRegards,\nGaurav Singh Yaadav"
        )

        if ok:
            st.success("✅ Test Email Sent Successfully.")
        else:
            st.error("❌ Email Sending Failed.")

    st.divider()

    st.caption(
        "NSE Scanner Authentication System v1.0"
    )
    

    st.divider()

    # =====================================================
    # SYSTEM INFORMATION
    # =====================================================

    st.caption(
        "NSE Scanner Authentication System v1.0"
    )

