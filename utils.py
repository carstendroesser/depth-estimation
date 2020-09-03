import smtplib
import ssl


def crop_center(array, crop_y, crop_x):
    if len(array.shape) != 4:
        raise Exception('Bad shape: input has to be a batch of 3D data')
    y = array.shape[1]
    x = array.shape[2]
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return array[:, start_y:start_y + crop_y, start_x:start_x + crop_x, :]


def concatenate_model_name(params):
    model_name = params[3] + '_' \
                 + params[4] + '_' + params[5] + '_' + params[6] + '_' + params[7] + '_' \
                 + params[8] + '_' \
                 + params[9] \
                 + '_e' + params[13] \
                 + '_bs' + params[14] \
                 + '_lr' + params[15].replace(".", "_") \
                 + '_' + params[16]
    return model_name


def send_update(message, sender_mail, sender_password, receiver_mail):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_mail, sender_password)
        server.sendmail(sender_mail, receiver_mail, message)
