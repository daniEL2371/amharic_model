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
    with open(full_name, "w") as json_  file:
        json_file.write(model_json)


def load_model(model_name): 
    with open(model_name, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        return load_model


def initializer(shape):
    return tf.random_uniform(shape, minval=-0.08, maxval=0.08)


def multi_task(input_shape, output_shapes, lstm=False, decay=0.0002):
    if lstm:
        CELL = LSTM
    else:
        CELL = GRU
    x = Input(shape=input_shape)
    z = CELL(256, return_sequences=True)(x)
    z = Dropout(0.2)(z)
    z = CELL(256, return_sequences=False)(z)
    z = Dropout(0.2)(z)
    # z = CELL(256, return_sequences=False)(z)
    # z = Dropout(0.5)(z)
    y_vowel = Dense(output_shapes[1],
                    activation="softmax", name="vowel_output")(z)
    y_cons = Dense(output_shapes[0],
                   activation="softmax", name="cons_output")(z)
    model = Model(inputs=x, output=[y_cons, y_vowel])
    adam = keras.optimizers.Adam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(loss="categorical_crossentropy",
                  optimizer=adam, metrics=['acc'])
    model.summary()
    return model


def get_model(input_shape, output_shape, lstm_cell=True, decay=0.0002):
    if lstm_cell:
        CELL = LSTM
    else:
        CELL = GRU
    model = Sequential()
    model.add(CELL(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(.5))
    model.add(CELL(256, return_sequences=True))
    model.add(Dropout(.5))
    model.add(CELL(256, return_sequences=False))
    model.add(Dropout(.5))
    model.add(Dense(output_shape, activation="softmax"))
    adam = keras.optimizers.Adam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['acc'])
    model.summary()
    return model


def load_model(model_name, model_tag):
    json = "{0}-{1}.json".format(model_name, model_tag)

def get_epoch(state):
    epoch = int(state.split('-')[1])
    return epoch

def get_model_wights(model_name, model_tag):
    folder = "model_weights/{0}_{1}/*.hdf5".format(model_name, model_tag)
    list_of_files = glob.glob(folder)
    list_of_files.sort(key=get_epoch)
    return list_of_files

def load_best_state(model_name, model_tag):
    folder = "model_weights/{0}_{1}/*.hdf5".format(model_name, model_tag)
    list_of_files = glob.glob(folder)
    list_of_files.sort(key=get_epoch)
    if len(list_of_files) == 0:
        return 0
    states = list_of_files
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
