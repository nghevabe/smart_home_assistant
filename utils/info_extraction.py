import re
from constants.word_set import *


def info_extractor(input_text):
    return get_action(input_text), get_room(input_text), get_color_code(input_text), get_song_name(input_text)


def get_room(input_text):
    index = 0
    for word in lst_word_room_name:
        if word in input_text:
            index = lst_word_room_name.index(word)
            # print(word)

    return lst_word_room_id[index]


def get_action(input_text):
    index = 0
    for word in lst_word_action_name:
        if word in input_text:
            index = lst_word_action_name.index(word)
            # print(word)

    return lst_word_action_id[index]


def get_color_code(input_text):
    index = 0
    for word in lst_word_color_name:
        if word in input_text:
            index = lst_word_color_name.index(word)
            # print(word)

    return lst_word_color_value[index]


def get_song_name(input_text):
    str_truncate = input_text
    for word in lst_word_music_action:
        s = re.sub(word, '', str_truncate)
        str_truncate = s

    for word in lst_word_action_name:
        s = re.sub(word, '', str_truncate)
        str_truncate = s

    for word in lst_word_stop_word:
        s = re.sub(word, '', str_truncate)
        str_truncate = s

    str_truncate = re.sub(r'\s{2}', ' ', str_truncate)
    str_truncate = re.sub(r'\s{3}', ' ', str_truncate)

    return str_truncate


def normal_str(input_text):
    str_normal = ""
    words = input_text.split(" ")

    for word in words:
        if word != '':
            str_normal = str_normal + " " + word

    return str_normal[1:]
