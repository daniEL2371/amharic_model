from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras import initializers
import keras
import tensorflow as tf
import glob
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
    class_model.add(CELL(128, input_shape=input_shape, return_sequences=True))
    class_model.add(CELL(128))
    class_model.add(Dense(output_shape, activation="softmax"))
    class_model.compile(loss="categorical_crossentropy", optimizer="adam")
    return class_model


def load_model(model_name, model_tag):
    json = "{0}-{1}.json".format(model_name, model_tag)


def load_best_state(model_name, model_tag):
    folder = "model_weights/{0}_{1}/*.hdf5".format(model_name, model_tag)
    list_of_files = glob.glob(folder)
    if len(list_of_files) == 0:
        return 0
    states = list(sorted(list_of_files))
    min_cost = 999
    k = 0
    for i, state in enumerate(states):
        cost = float(state[-11:-5])
        if cost < min_cost:
            min_cost = cost
            k = i
    best_state = states[k]
    print("Loading State: " + best_state)
    model = keras.models.load_model(best_state)
    return model
