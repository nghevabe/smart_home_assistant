import re
import requests
from bs4 import BeautifulSoup


def get_mp3_url_song(song_name):
    query_name = encode_query(song_name)
    response_search = res_search_query(query_name)
    first_result = get_first_result(response_search)
    play_song_url = get_play_song_url(first_result)
    download_song_url = get_download_song_url(play_song_url)
    return download_song_url


def encode_query(song_name):
    song_encoded = re.sub(r' ', '+', song_name)
    return song_encoded.lower()


def res_search_query(song_name):
    request_url = "https://www.nhaccuatui.com/tim-kiem?q=" + song_name
    resp = requests.get(request_url)
    return resp.text


def get_first_result(content):
    soup = BeautifulSoup(content, 'html5lib')
    paragraphs = soup.find('li', attrs={'class': 'sn_search_single_song'})
    return paragraphs.a['href']


def get_play_song_url(url):
    song_detail_data = requests.get(url)
    link_song = re.findall(r'".+xml[?].+"', song_detail_data.text)
    return link_song[0][1:-1]


def get_download_song_url(play_song_url):
    song_download_data = requests.get(play_song_url)
    url_download = re.findall(r'https.+.mp3.+]]>', song_download_data.text)
    return url_download[0][:-3]


# print(get_mp3_url_song("Đi giữa trời rực rỡ"))
