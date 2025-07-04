import keras
from utils.pre_processing import *


def predictions(texts, my_model):
    from nltk import word_tokenize
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

    return pred


def get_final_output(predicts):
    prediction = predicts[0]

    classes = np.array(unique_intent)
    ids = np.argsort(-prediction)
    classes = classes[ids]
    prediction = -np.sort(-prediction)

    for i in range(predicts.shape[1]):
        print("%s has confidence = %s" % (classes[i], (prediction[i])))


def get_best_result(predicts):
    prediction = predicts[0]
    classes = np.array(unique_intent)
    ids = np.argsort(-prediction)
    classes = classes[ids]

    return classes[0]


def get_result_request(request):
    loaded_model = keras.models.load_model(os.getcwd() + "/model/my_model.h5")
    text = request
    predict = predictions(text, loaded_model)
    return get_best_result(predict)


# if __name__ == '__prediction__':
#     print(os.getcwd())
#     os.chdir('../')
#     print(os.getcwd())
#
#     intent, unique_intent, sentences = load_dataset(os.getcwd() + "/data/my_data.csv")
#     print(sentences[:5])
#
#     cleaned_words = cleaning(sentences)
#     print(len(cleaned_words))
#     print(cleaned_words[:2])
#
#     word_tokenizer = create_tokenizer(cleaned_words)
#     vocab_size = len(word_tokenizer.word_index) + 1
#     max_length = max_length(cleaned_words)
