# ==========================================
# database.py
# SQLite Database Functions
# ==========================================

import sqlite3
from datetime import datetime, timedelta

from config import (
    DATABASE_PATH,
    ADMIN_ROLE,
    USER_ROLE,
    STATUS_PENDING,
    STATUS_APPROVED,
    STATUS_DISABLED,
    DEFAULT_SUBSCRIPTION_DAYS,
)

# ==========================================
# CONNECT DATABASE
# ==========================================

def get_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ==========================================
# CREATE TABLE
# ==========================================

def create_database():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT UNIQUE,

        password TEXT,

        fullname TEXT,

        mobile TEXT,

        email TEXT,

        role TEXT,

        status TEXT,

        expiry_date TEXT,

        created_on TEXT

    )
    """)

    conn.commit()
    conn.close()


# ==========================================
# CREATE DEFAULT ADMIN
# Password should already be HASHED
# ==========================================

def create_admin(username, hashed_password):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    if cur.fetchone() is None:

        expiry = (
            datetime.now() +
            timedelta(days=3650)
        ).strftime("%Y-%m-%d")

        cur.execute("""
        INSERT INTO users
        (
            username,
            password,
            fullname,
            mobile,
            email,
            role,
            status,
            expiry_date,
            created_on
        )

        VALUES (?,?,?,?,?,?,?,?,?)
        """,

        (
            username,
            hashed_password,
            "Administrator",
            "",
            "",
            ADMIN_ROLE,
            STATUS_APPROVED,
            expiry,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()

    conn.close()


# ==========================================
# REGISTER USER
# ==========================================

def register_user(
        username,
        password,
        fullname,
        mobile,
        email
):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    if cur.fetchone():

        conn.close()

        return False

    expiry = (
        datetime.now() +
        timedelta(days=DEFAULT_SUBSCRIPTION_DAYS)
    ).strftime("%Y-%m-%d")

    cur.execute("""

    INSERT INTO users(

    username,
    password,
    fullname,
    mobile,
    email,
    role,
    status,
    expiry_date,
    created_on

    )

    VALUES(?,?,?,?,?,?,?,?,?)

    """,

    (

        username,
        password,
        fullname,
        mobile,
        email,
        USER_ROLE,
        STATUS_PENDING,
        expiry,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ))

    conn.commit()
    conn.close()

    return True


# ==========================================
# LOGIN USER
# ==========================================

def get_user(username):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    row = cur.fetchone()

    conn.close()

    return row


# ==========================================
# GET ALL USERS
# ==========================================

def get_all_users():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users ORDER BY id DESC"
    )

    rows = cur.fetchall()

    conn.close()

    return rows


# ==========================================
# PENDING USERS
# ==========================================

def get_pending_users():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE status=?",
        (STATUS_PENDING,)
    )

    rows = cur.fetchall()

    conn.close()

    return rows


# ==========================================
# APPROVE USER
# ==========================================

def approve_user(user_id):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        UPDATE users

        SET status=?

        WHERE id=?

        """,

        (

            STATUS_APPROVED,
            user_id

        )

    )

    conn.commit()

    conn.close()


# ==========================================
# DISABLE USER
# ==========================================

def disable_user(user_id):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        UPDATE users

        SET status=?

        WHERE id=?

        """,

        (

            STATUS_DISABLED,
            user_id

        )

    )

    conn.commit()

    conn.close()


# ==========================================
# DELETE USER
# ==========================================

def delete_user(user_id):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        "DELETE FROM users WHERE id=?",

        (

            user_id,

        )

    )

    conn.commit()

    conn.close()


# ==========================================
# EXTEND SUBSCRIPTION
# ==========================================

def extend_subscription(user_id, days):

    conn = get_connection()

    cur = conn.cursor()

    expiry = (
        datetime.now() +
        timedelta(days=days)
    ).strftime("%Y-%m-%d")

    cur.execute(

        """

        UPDATE users

        SET expiry_date=?

        WHERE id=?

        """,

        (

            expiry,

            user_id

        )

    )

    conn.commit()

    conn.close()


# ==========================================
# CHANGE PASSWORD
# ==========================================

def update_password(username, hashed_password):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        UPDATE users

        SET password=?

        WHERE username=?

        """,

        (

            hashed_password,

            username

        )

    )

    conn.commit()

    conn.close()
