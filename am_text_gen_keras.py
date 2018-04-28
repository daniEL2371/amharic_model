import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os
from am_process_text import DataGen

seq_length = 100
batch_size = 200
hidden_size = 64

print("Generating Data")
gen = DataGen("data/small.txt", "data/geez.txt", batch_size, seq_length)
gen.process()
print("Total Training Data: {0}".format(len(gen.train_X)))
x_dims, y_dims, z_dims = gen.data_dims

epoches = 20
batches = len(gen.train_X) // batch_size
iterations = epoches * batches
print("Batches: {0}, Epoches: {1}".format(batches, epoches))

model1 = Sequential()
outputs, state_h, state_c = LSTM(256, input_shape=(
    gen.train_X.shape[1], gen.train_X.shape[2]), return_state=True)
model1.add(outputs)
model1.add(Dense(gen.train_Y_classes.shape[1], activation="softmax"))
model1.compile(loss="categorical_crossentropy", optimizer="adam")

model2 = Sequential()
model2.add(LSTM(256, input_shape=(gen.train_X.shape[1], gen.train_X.shape[2])))
model2.add(Dense(gen.train_Y_chars.shape[1], activation="softmax"))
model2.compile(loss="categorical_crossentropy", optimizer="adam")

# model1.fit(gen.train_X, gen.train_Y_classes, epochs=5, batch_size=64)
model2.fit(gen.train_X, gen.train_Y_chars, epochs=5, batch_size=64)
