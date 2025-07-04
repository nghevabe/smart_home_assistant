import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import re
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from sklearn.model_selection import train_test_split



def load_dataset(filenames):
    df = pd.read_csv(filenames, encoding="latin1", names=["Sentence", "Intent"])
    print(df.head())
    intents = df["Intent"]
    unique_intents = list(set(intents))
    sentencess = list(df["Sentence"])

    print("unique_intents")
    print(unique_intents)
    print("unique_intents")

    return intents, unique_intents, sentencess


intent, unique_intent, sentences = load_dataset("data_set.csv")
print(sentences[:5])

# define stemmer
stemmer = LancasterStemmer()


def cleaning(sen):
    words = []
    for s in sen:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
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
    new_model.add(Embedding(vocabs_size, 128, input_length=max_lengths, trainable=False))
    new_model.add(Bidirectional(LSTM(128)))
    new_model.add(Dense(32, activation="relu"))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(21, activation="softmax"))

    return new_model


model = create_model(vocab_size, max_length)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_X, train_Y, epochs=80, batch_size=16, validation_data=(val_X, val_Y))


def predictions(texts, my_model):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", texts)
    test_word = word_tokenize(clean)
    test_word = [w.lower() for w in test_word]
    test_ls = word_tokenizer.texts_to_sequences(test_word)
    print(test_word)
    # Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))

    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    x = padding_doc(test_ls, max_length)
    print("padding_doc")
    print(x)
    pred = my_model.predict(x)

    return pred


def get_final_output(predicts, classes):
    prediction = predicts[0]

    classes = np.array(classes)
    ids = np.argsort(-prediction)
    classes = classes[ids]
    prediction = -np.sort(-prediction)

    for i in range(predicts.shape[1]):
        print("%s has confidence = %s" % (classes[i], (prediction[i])))


# text = "To which types of businesses do you offer loan"
# predict = predictions(text, model)
# get_final_output(predict, unique_intent)
