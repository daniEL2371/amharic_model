from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras import initializers
import tensorflow as tf
import os


def save_model(model, name, tag):
    cwd = os.getcwd()
    full_name = "{0}/models/{1}-{2}.json".format(cwd, name, tag)
    model_json = model.to_json()
    with open(full_name, "w") as json_file:
        json_file.write(model_json)


def load_model(model_name):
    with open(model_name, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        return load_model


def initializer(shape):
    return tf.random_uniform(shape, minval=-0.08, maxval=0.08)


def get_model(input_shape, output_shape, lstm_cell=True):
    if lstm_cell:
        CELL = LSTM
    else:
        CELL = GRU
    class_model = Sequential()
    class_model.add(CELL(256, input_shape=input_shape,
                         return_sequences=True, 
                         kernel_initializer=initializer, 
                         recurrent_initializer=initializer))
    class_model.add(Dropout(.4))
    class_model.add(CELL(256, kernel_initializer=initializer,
                         recurrent_initializer=initializer))
    class_model.add(Dense(output_shape, activation="softmax",
                          kernel_initializer=initializer))
    rms = optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=None, decay=0.99)
    class_model.compile(loss="categorical_crossentropy", optimizer=rms)
    return class_model
