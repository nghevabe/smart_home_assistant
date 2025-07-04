from firebase_admin import db


def send_signal(room, device, lst_color_code, status):
    node = db.reference('smart_home_assistant/home_device/'+room+'/'+device)
    if 'light' in device:
        node.update({
            'red': lst_color_code[0],
            'green': lst_color_code[1],
            'blue': lst_color_code[2],
        })
    else:
        node.update({
            'status': status
        })
