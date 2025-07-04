import keras
import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from sklearn.model_selection import train_test_split
from keras.layers import Dense, MaxPooling2D, Flatten, Convolution2D
from unidecode import unidecode
import firebase_admin
from firebase_admin import db, credentials

# from tensorflow.python.keras.layers import GRU


is_crop = 0
is_violated = 0

# Firebase
# def listener(event):
#     global is_violated
#     print(event.event_type)  # can be 'put' or 'patch'
#     print(event.path)  # relative to the reference, it seems
#     print(event.data)  # new data at /reference/event.path. None if deleted
#     if event.data == "1":
#         is_violated = 1
#     else:
#         is_violated = 0
#
# json_path = r'C:\Users\Public\cred.json'
#
# cred = credentials.Certificate(json_path)
# obj = firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://iot-server-edaa9-default-rtdb.firebaseio.com'
# })
#
# db.reference('security_system/sensor_metal_detect', app=obj).listen(listener)
# Firebase


def load_dataset(filenames):
    df = pd.read_csv(filenames, encoding="utf-8", names=["Sentence", "Intent"])
    print(df.head())
    intents = df["Intent"]
    unique_intents = list(set(intents))
    unique_intents.sort()
    sentencess = list(df["Sentence"])
    print("unique_intents_sorted")
    print(unique_intents)

    return intents, unique_intents, sentencess


intent, unique_intent, sentences = load_dataset("my_data.csv")
print(sentences[:5])


def cleaning(sen):
    words = []

    for s in sen:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", unidecode(s))
        w = word_tokenize(clean)
        # stemming
        words.append([i.lower() for i in w])

    return words


cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])


def create_tokenizer(words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token


def max_length(words):
    return len(max(words, key=len))


word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))


def encoding_doc(token, words):
    return token.texts_to_sequences(words)


encoded_doc = encoding_doc(word_tokenizer, cleaned_words)


def padding_doc(encoded_docs, max_lengths):
    return pad_sequences(encoded_docs, maxlen=max_lengths, padding="post")


padded_doc = padding_doc(encoded_doc, max_length)

var = padded_doc[:5]

print("Shape of padded docs = ", padded_doc.shape)

# tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~')

encoded_output = encoding_doc(output_tokenizer, intent)
encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)


def one_hot(encode):
    o = OneHotEncoder(sparse_output=False)
    return o.fit_transform(encode)


output_one_hot = one_hot(encoded_output)

train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle=True, test_size=0.2)

print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


def create_model(vocabs_size, max_lengths):
    new_model = Sequential()
    new_model.add(Embedding(vocabs_size, 32, input_length=max_lengths, trainable=False))
    new_model.add(Bidirectional(LSTM(32)))
    new_model.add(Dense(16, activation="relu"))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(3, activation="softmax"))

    return new_model


# model = create_model(vocab_size, max_length)
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(train_X, train_Y, epochs=50, batch_size=16, validation_data=(val_X, val_Y))
# model.save('my_model.h5')


def predictions(texts, my_model):
    test_word = word_tokenize(unidecode(texts))
    print("test_word")
    print(test_word)
    test_word = [w.lower() for w in test_word]
    test_ls = word_tokenizer.texts_to_sequences(test_word)
    # Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))

    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    x = padding_doc(test_ls, max_length)
    pred = my_model.predict(x)
    print("predict")
    print(pred)

    return pred


def get_final_output(predicts, classes):
    prediction = predicts[0]

    classes = np.array(classes)
    ids = np.argsort(-prediction)
    classes = classes[ids]
    prediction = -np.sort(-prediction)

    for i in range(predicts.shape[1]):
        print("%s has confidence = %s" % (classes[i], (prediction[i])))


loaded_model = keras.models.load_model('model/my_model.h5')
text = "Đóng hộ tôi cái cửa ở phòng họp"
predict = predictions(text, loaded_model)
get_final_output(predict, unique_intent)

# Firebase
# if is_violated == 1:
#     crop_frame = frame[bb[i][1]:bb[i][3] + 25, bb[i][0]:bb[i][2] + 20, :]
#     directory = r'C:\Users\Linh Tran\PycharmProjects\MiAI_FaceRecog_3\people_violate'
#     os.chdir(directory)
#     str_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     cv2.imwrite(best_name + "_" + str_time + ".png", crop_frame)
#
#     node = db.reference('security_system')
#     node.update({
#         'alarm_device': '1'
#     })
#
#     time.sleep(10)
#
#     node.update({
#         'alarm_device': '0',
#     })
#     node.update({
#         'sensor_metal_detect': '0'
#     })
#
#     is_violated = 0
# Firebase