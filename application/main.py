from crawler.data_crawl_process import get_mp3_url_song
from utils.info_extraction import info_extractor, normal_str
from utils.iot_process import send_signal
from utils.prediction import get_result_request
import firebase_admin
from firebase_admin import db, credentials


def get_intent_info(input_text):
    intent = get_result_request(input_text)
    status, room, color_code, song_name = info_extractor(input_text.lower())
    return intent, status, room, color_code, song_name


def listener(event):
    print("data from firebase:")
    print(event.data)  # new data at /reference/event.path. None if deleted
    # get_result_request(event.data)
    # device, status, room, color_code, song_name = get_intent_info("Đóng cửa phòng khách cho tôi")
    intent, status, room, color_code, song_name = get_intent_info(event.data)
    print("intent: ", intent)
    if intent == 'music':
        print("song_name: ", normal_str(song_name))
        print("url: ", get_mp3_url_song(normal_str(song_name)))
    else:
        send_signal(room, intent, color_code, status)
        print("status: ", status)
        print("room: ", room)
        print("color_code: ", color_code)


json_path = r'C:\Users\Public\cred.json'

cred = credentials.Certificate(json_path)
obj = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-server-edaa9-default-rtdb.firebaseio.com'
})

# Bật đèn phòng khách
# Chuyển đèn ở phòng khách sang màu xanh cho tôi
if __name__ == '__main__':
    # smart_home_assistant/home_device/linhth_house/device/light_living_room/blue
    db.reference('smart_home_assistant/virtual_assistant/request', app=obj).listen(listener)






