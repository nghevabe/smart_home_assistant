from utils.pre_processing import *
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Embedding, Dropout
from sklearn.model_selection import train_test_split
from keras.layers import Dense


def create_model(vocabs_size, max_lengths):
    new_model = Sequential()
    new_model.add(Embedding(vocabs_size, 32, input_length=max_lengths, trainable=False))
    new_model.add(Bidirectional(LSTM(32)))
    new_model.add(Dense(16, activation="relu"))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(3, activation="softmax"))

    return new_model


def training_model():
    train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle=True, test_size=0.2)

    print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
    print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))

    model = create_model(vocab_size, max_length)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_X, train_Y, epochs=50, batch_size=16, validation_data=(val_X, val_Y))
    # os.chdir('../')
    model.save(os.getcwd() + "/model/my_model_test.h5")






