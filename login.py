import streamlit as st

from login import login_page
from admin import admin_panel
from database import create_database
from register import register_page

# Create DB Automatically
create_database()

st.set_page_config(
    page_title="NSE Scanner",
    page_icon="📈",
    layout="wide"
)

# Session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login
if not st.session_state.authenticated:

    login_page()

    st.stop()

# Admin Panel
if st.session_state.role == "Admin":

    admin_panel()

# -----------------------------
# Your Scanner Starts Here
# -----------------------------

st.title("📈 NSE Scanner")
