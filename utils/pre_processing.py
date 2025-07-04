import os

import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import re
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from unidecode import unidecode


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


def cleaning(sen):
    words = []

    for s in sen:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", unidecode(s))
        w = word_tokenize(clean)
        # stemming
        words.append([i.lower() for i in w])

    return words


def create_tokenizer(words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token


def max_length(words):
    return len(max(words, key=len))


def encoding_doc(token, words):
    return token.texts_to_sequences(words)


def padding_doc(encoded_docs, max_lengths):
    return pad_sequences(encoded_docs, maxlen=max_lengths, padding="post")


def one_hot(encode):
    o = OneHotEncoder(sparse_output=False)
    return o.fit_transform(encode)


os.chdir('../')
intent, unique_intent, sentences = load_dataset(os.getcwd() + "/data/my_data.csv")
cleaned_words = cleaning(sentences)
word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
padded_doc = padding_doc(encoded_doc, max_length)
var = padded_doc[:5]
print("Shape of padded docs = ", padded_doc.shape)

# tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~')

encoded_output = encoding_doc(output_tokenizer, intent)
encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)

output_one_hot = one_hot(encoded_output)



