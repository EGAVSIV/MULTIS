# ==========================================
# styles.py
# Custom Streamlit Styles
# ==========================================

import streamlit as st


def load_css():

    st.markdown("""
    <style>

    /* ----------------------------
       Hide Streamlit Default Menu
    -----------------------------*/

    #MainMenu {
        visibility:hidden;
    }

    footer {
        visibility:hidden;
    }

    header {
        visibility:hidden;
    }

    /* ----------------------------
       Main App
    -----------------------------*/

    .main{
        padding-top:1rem;
    }

    /* ----------------------------
       Buttons
    -----------------------------*/

    .stButton>button{

        width:100%;

        border-radius:10px;

        height:45px;

        font-size:16px;

        font-weight:600;

        border:none;

    }

    /* ----------------------------
       Textbox
    -----------------------------*/

    .stTextInput>div>div>input{

        border-radius:8px;

    }

    /* ----------------------------
       Sidebar
    -----------------------------*/

    section[data-testid="stSidebar"]{

        width:300px;

    }

    /* ----------------------------
       Metric Cards
    -----------------------------*/

    div[data-testid="metric-container"]{

        border:1px solid #dddddd;

        border-radius:12px;

        padding:15px;

        box-shadow:0px 2px 6px rgba(0,0,0,.15);

    }

    /* ----------------------------
       Expander
    -----------------------------*/

    .streamlit-expanderHeader{

        font-size:17px;

        font-weight:bold;

    }

    /* ----------------------------
       Success Message
    -----------------------------*/

    div[data-testid="stAlert"]{

        border-radius:10px;

    }

    /* ----------------------------
       Login Card
    -----------------------------*/

    .login-box{

        background:#262730;

        padding:25px;

        border-radius:15px;

        border:1px solid #444;

    }

    /* ----------------------------
       Footer
    -----------------------------*/

    .footer{

        text-align:center;

        color:gray;

        font-size:13px;

        margin-top:30px;

    }

    </style>
    """, unsafe_allow_html=True)


# ==========================================
# Footer
# ==========================================

def show_footer():

    st.markdown("""

    <div class='footer'>

    © 2026 Gaurav Singh Yaadav

    <br>

    NSE Stock Market Scanner

    </div>

    """, unsafe_allow_html=True)
