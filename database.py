# ==========================================
# database.py (Part-1)
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
    DEFAULT_SUBSCRIPTION_DAYS
)

from utils import hash_password


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_connection():

    conn = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    return conn


# ==========================================
# CREATE DATABASE
# ==========================================

def create_database():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""

    CREATE TABLE IF NOT EXISTS users(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT UNIQUE NOT NULL,

        password TEXT NOT NULL,

        fullname TEXT,

        mobile TEXT,

        email TEXT,

        role TEXT,

        status TEXT,

        expiry_date TEXT,

        created_on TEXT

    )

    """)

    # ---------------------------------
    # Create Default Admin
    # ---------------------------------

    cur.execute(

        "SELECT * FROM users WHERE username=?",

        ("admin",)

    )

    admin = cur.fetchone()

    if admin is None:

        password = hash_password("admin123")

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

            "admin",

            password,

            "Administrator",

            "",

            "",

            ADMIN_ROLE,

            STATUS_APPROVED,

            "2099-12-31",

            datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ))

        print("Admin Created")

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

        datetime.now()

        + timedelta(days=DEFAULT_SUBSCRIPTION_DAYS)

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

    VALUES(

        ?,?,?,?,?,?,?,?,?

    )

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
# GET USER
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
# GET PENDING USERS
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
        (user_id,)
    )

    conn.commit()
    conn.close()


# ==========================================
# UPDATE PASSWORD
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


# ==========================================
# EXTEND SUBSCRIPTION
# ==========================================

def extend_subscription(user_id, days):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT expiry_date FROM users WHERE id=?",
        (user_id,)
    )

    row = cur.fetchone()

    if row:

        try:

            current_expiry = datetime.strptime(
                row["expiry_date"],
                "%Y-%m-%d"
            )

            # If already expired, start from today
            if current_expiry < datetime.now():

                current_expiry = datetime.now()

            new_expiry = current_expiry + timedelta(days=days)

            cur.execute(
                """
                UPDATE users
                SET expiry_date=?
                WHERE id=?
                """,
                (
                    new_expiry.strftime("%Y-%m-%d"),
                    user_id
                )
            )

            conn.commit()

        except Exception as e:

            print(e)

    conn.close()


# ==========================================
# GET USER BY ID
# ==========================================

def get_user_by_id(user_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE id=?",
        (user_id,)
    )

    row = cur.fetchone()

    conn.close()

    return row


# ==========================================
# UPDATE PROFILE
# ==========================================

def update_profile(
    user_id,
    fullname,
    mobile,
    email
):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE users

        SET

        fullname=?,
        mobile=?,
        email=?

        WHERE id=?
        """,

        (
            fullname,
            mobile,
            email,
            user_id
        )
    )

    conn.commit()
    conn.close()


# ==========================================
# USER EXISTS
# ==========================================

def user_exists(username):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM users WHERE username=?",
        (username,)
    )

    exists = cur.fetchone() is not None

    conn.close()

    return exists


# ==========================================
# USER COUNT
# ==========================================

def get_user_count():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM users"
    )

    count = cur.fetchone()[0]

    conn.close()

    return count


# ==========================================
# PENDING COUNT
# ==========================================

def get_pending_count():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM users WHERE status=?",
        (STATUS_PENDING,)
    )

    count = cur.fetchone()[0]

    conn.close()

    return count


# ==========================================
# APPROVED COUNT
# ==========================================

def get_approved_count():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM users WHERE status=?",
        (STATUS_APPROVED,)
    )

    count = cur.fetchone()[0]

    conn.close()

    return count


# ==========================================
# DISABLED COUNT
# ==========================================

def get_disabled_count():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM users WHERE status=?",
        (STATUS_DISABLED,)
    )

    count = cur.fetchone()[0]

    conn.close()

    return count
