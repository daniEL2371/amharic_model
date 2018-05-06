from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model


def get_class_model(input_shape, output_shape):
    class_model = Sequential()
    class_model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    class_model.add(Dropout(.2))
    class_model.add(LSTM(256))
    class_model.add(Dense(output_shape, activation="softmax"))
    class_model.compile(loss="categorical_crossentropy", optimizer="adam")
    return class_model


def get_char_model(input_shape, output_shape):
    char_model = Sequential()
    char_model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    char_model.add(Dropout(.2))
    char_model.add(LSTM(256))
    char_model.add(Dense(output_shape, activation="softmax"))
    char_model.compile(loss="categorical_crossentropy", optimizer="adam")
    return char_model
