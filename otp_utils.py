# ==========================================
# otp_utils.py
# Author : Gaurav Singh Yaadav
# ==========================================

import random
import string
from datetime import datetime, timedelta

from config import OTP_LENGTH, OTP_VALIDITY_MINUTES


# ==========================================
# GENERATE OTP
# ==========================================

def generate_otp():

    return "".join(

        random.choices(

            string.digits,

            k=OTP_LENGTH

        )

    )


# ==========================================
# OTP EXPIRY
# ==========================================

def otp_expiry_time():

    return (

        datetime.now() +

        timedelta(

            minutes=OTP_VALIDITY_MINUTES

        )

    ).strftime("%Y-%m-%d %H:%M:%S")


# ==========================================
# OTP VALID ?
# ==========================================

def is_otp_valid(expiry_time):

    try:

        expiry = datetime.strptime(

            expiry_time,

            "%Y-%m-%d %H:%M:%S"

        )

        return datetime.now() <= expiry

    except:

        return False


# ==========================================
# RANDOM PASSWORD
# ==========================================

def generate_password(length=10):

    chars = (

        string.ascii_letters +

        string.digits +

        "@#$%&"

    )

    return "".join(

        random.choice(chars)

        for _ in range(length)

    )
