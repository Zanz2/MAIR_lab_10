from __future__ import print_function
from playsound import playsound
from email.message import EmailMessage
from datetime import datetime
from smtpd import SMTPServer
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pyautogui
import webbrowser
import smtplib
import asyncore

def play_sound(filename):
    playsound(filename)

def press_button_on_users_keyboard(button):
    pyautogui.keyDown(button)
    pyautogui.keyUp(button)

def fill_google_form(data):
    webbrowser.open(
        'https://docs.google.com/forms/d/e/1FAIpQLSde0vdT4hmwuuIXd_BV_hOKTqoRxGLVRwpWE460UIXGMDLNpg/viewform?usp=pp_url&entry.1591633300=Comments&entry.326955045=aaaa&entry.1696159737=' + data["name"] + '+&entry.485428648=aaaaaaa&entry.879531967=' + data["email"] + ''
    )


def send_test_results_email(results, receiver_address):
    # The mail addresses and password
    sender_address = 'mair.group10@gmail.com'
    sender_pass = 'xxxxx'  # ask me for this irl :D, so its not visible on github
    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'A test mail sent by Python. It has an attachment.'  # The subject line
    # The body and the attachments for the mail
    message.attach(MIMEText(results, 'plain'))
    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')


#play_sound("discord_message.mp3")
#press_button_on_users_keyboard("win") # press users windows button to irritate them, :D
data = {
    "name": "user",
    "email": "zagarzan4@gmail.com"
}
#fill_google_form(data)

mail_content = '''Hello,
This is a simple mail. There is only text, no attachments are there The mail is sent using Python SMTP library.
Thank You'''
send_test_results_email(mail_content, receiver_address='put your email here') # ask me for the pass irl