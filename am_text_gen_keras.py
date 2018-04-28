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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from am_process_text import DataGen

seq_length = 100
batch_size = 200
hidden_size = 128

print("Generating Data")
gen = DataGen("data/small.txt", "data/geez.txt", batch_size, seq_length)
gen.process()
print("Total Training Data: {0}".format(len(gen.train_X)))
x_dims, y_dims, z_dims = gen.data_dims

input_shape = (x_dims[1], x_dims[2])
class_model = Sequential()
class_model.add(
    LSTM(hidden_size, input_shape=input_shape, return_sequences=True))
class_model.add(Dropout(.2))
class_model.add(LSTM(hidden_size))
class_model.add(Dense(gen.train_Y_classes.shape[1], activation="softmax"))
class_model.compile(loss="categorical_crossentropy", optimizer="adam")

char_model = Sequential()
char_model.add(LSTM(hidden_size, input_shape=(x_dims[1], x_dims[2])))
# char_model.add(LSTM(64))
char_model.add(Dense(gen.train_Y_chars.shape[1], activation="softmax"))
char_model.compile(loss="categorical_crossentropy", optimizer="adam")

class_model.fit(gen.train_X, gen.train_Y_classes, epochs=20, batch_size=200)
# class_model_json = class_model.to_json()
# with open("class_model.json", "w") as json_file:
#     json_file.write(class_model_json)
# # serialize weights to HDF5
class_model.save_weights("models/class_model-5.h5")

char_model.fit(gen.train_X, gen.train_Y_chars, epochs=20, batch_size=200)
# char_model_json = class_model.to_json()
# with open("class_model.json", "w") as json_file:
#     json_file.write(char_model_json)
# # serialize weights to HDF5
char_model.save_weights("models/char_model-5.h5")


class_model.load_weights("models/class_model-5.h5")
char_model.load_weights("models/char_model-5.h5")
show_len = 100
start = 200
seed_text = gen.raw_datatset[start:start + seq_length]
gen_seq = []
print(seed_text)
seed_text = list(seed_text)
for sh in range(show_len):

    seed_int = [gen.char2int[s] for s in seed_text]
    seedvec = np.array(seed_int, dtype=np.float32).reshape(
        (1, seq_length, 1)) / 350
    cs = class_model.predict(seedvec)[0].argmax() * 10
    cr = char_model.predict(seedvec)[0].argmax()
    char_int = cs + cr
    if char_int not in gen.int2char:
        char_int = 350
    gen_seq.append(char_int)
    seed_text.append(gen.int2char[char_int])
    seed_text = seed_text[1:]

gen_seq = ''.join([gen.int2char[g] for g in gen_seq])
print(gen_seq)
