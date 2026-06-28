# ==========================================
# admin.py
# Admin Dashboard
# ==========================================
st.title("👨‍💼 ADMIN PANEL")
st.success("Admin Panel Loaded")
import streamlit as st
import pandas as pd

from database import (
    get_all_users,
    approve_user,
    disable_user,
    delete_user,
    extend_subscription
)

from utils import logout


# ==========================================
# ADMIN PANEL
# ==========================================

def admin_panel():

    st.sidebar.success("👨‍💼 Admin")

    if st.sidebar.button("🚪 Logout"):
        logout()

    st.title("👨‍💼 Admin Dashboard")

    users = get_all_users()

    if len(users) == 0:

        st.info("No users found.")

        return

    # ------------------------------------
    # Dashboard
    # ------------------------------------

    total = len(users)

    approved = sum(
        1 for u in users if u["status"] == "Approved"
    )

    pending = sum(
        1 for u in users if u["status"] == "Pending"
    )

    disabled = sum(
        1 for u in users if u["status"] == "Disabled"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Users", total)
    c2.metric("Approved", approved)
    c3.metric("Pending", pending)
    c4.metric("Disabled", disabled)

    st.divider()

    # ------------------------------------
    # User List
    # ------------------------------------

    st.subheader("Registered Users")

    for user in users:

        with st.expander(
            f"{user['fullname']} ({user['username']})"
        ):

            st.write(f"**ID :** {user['id']}")
            st.write(f"**Username :** {user['username']}")
            st.write(f"**Full Name :** {user['fullname']}")
            st.write(f"**Mobile :** {user['mobile']}")
            st.write(f"**Email :** {user['email']}")
            st.write(f"**Role :** {user['role']}")
            st.write(f"**Status :** {user['status']}")
            st.write(f"**Expiry :** {user['expiry_date']}")

            col1, col2, col3, col4 = st.columns(4)

            # ------------------------------
            # Approve
            # ------------------------------

            with col1:

                if st.button(
                    "✅ Approve",
                    key=f"a{user['id']}"
                ):

                    approve_user(user["id"])

                    st.success("User Approved")

                    st.rerun()

            # ------------------------------
            # Disable
            # ------------------------------

            with col2:

                if st.button(
                    "🚫 Disable",
                    key=f"d{user['id']}"
                ):

                    disable_user(user["id"])

                    st.warning("User Disabled")

                    st.rerun()

            # ------------------------------
            # Extend Subscription
            # ------------------------------

            with col3:

                if st.button(
                    "📅 +30 Days",
                    key=f"e{user['id']}"
                ):

                    extend_subscription(
                        user["id"],
                        30
                    )

                    st.success("Subscription Extended")

                    st.rerun()

            # ------------------------------
            # Delete User
            # ------------------------------

            with col4:

                if user["role"] != "Admin":

                    if st.button(
                        "🗑 Delete",
                        key=f"x{user['id']}"
                    ):

                        delete_user(user["id"])

                        st.error("User Deleted")

                        st.rerun()

    st.divider()

    # ------------------------------------
    # Export User List
    # ------------------------------------

    st.subheader("Export Users")

    df = pd.DataFrame(
        [dict(u) for u in users]
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download CSV",
        csv,
        file_name="users.csv",
        mime="text/csv"
    )
