from email_service import send_email

ok = send_email(

    "nse.scanner.app@gmail.com",

    "Testing Email",

    "Congratulations! Email Service is Working."

)

print(ok)
