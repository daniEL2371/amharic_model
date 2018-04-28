import numpy as np
from util import *
import os


class DataGen:

    def __init__(self, dataset_file, geez_file, batch_size, seuqnce_length):
        self.dataset_filename = dataset_file
        self.geez_chars_filename = geez_file
        self.raw_datatset = ""
        self.geez_chars = None
        self.vocab = None
        self.char2int = {}
        self.char2tup = {}
        self.int2char = {}
        self.train_X = None
        self.train_Y_classes = None
        self.train_Y_chars = None
        self.batch = 0
        self.iterator = None
        self.batch_size = batch_size
        self.seuqnce_length = seuqnce_length
        self.data_dims = None
        self.data_file = open(dataset_file, mode='r', encoding='utf-8')
        self.total_num_chars = 14908472
        self.current_read_chars = 0

    def geez_to_dict(self):
        data = open(self.geez_chars_filename, encoding='utf-8').readlines()
        l = []
        for line in data:
            f = []
            for c in line:
                if not c.isspace():
                    f.append(c)
            if len(f) > 0:
                l.append(f)

        k = 0
        char2int = {}
        int2char = {}
        char2tup = {}
        for k in range(34):
            row = l[k]
            for i in range(len(row)):
                char2int[row[i]] = k * 10 + i
                int2char[k * 10 + i] = row[i]
                char2tup[row[i]] = (k * 10, i)

        for k in range(34, 37):
            row = l[k]
            for i in range(len(row)):
                char2int[row[i]] = 340
                int2char[340] = row[i]
                char2tup[row[i]] = (34, 0)
        char2int[' '] = 350
        int2char[350] = ' '
        char2tup[' '] = (350, 0)

        self.int2char = int2char
        self.char2int = char2int
        self.geez_chars = char2tup
        self.char2tup = char2tup

    def read_dataset(self):
        self.raw_datatset = open(
            self.dataset_filename, encoding="utf-8").read()

    def chars_to_dict(self):
        chars = set(self.raw_datatset)
        for i, c in enumerate(chars):
            if c in self.geez_chars:
                self.char2tup[c] = self.geez_chars[c]
                index = self.geez_chars[c][0] + self.geez_chars[c][1]
                self.int2char[index] = c
                self.char2int[c] = index
        self.vocab = list(self.char2int.keys())

    def prepare_training_data(self):
        seuqnce_length = self.seuqnce_length
        int_encoded = [self.char2int[c] for c in self.raw_datatset]
        class_output_size = 36
        char_output_size = 7
        N_DATA = len(self.raw_datatset) - seuqnce_length - 1
        data = np.empty((N_DATA, seuqnce_length, 1), dtype=np.float32)
        self.train_Y_classes = np.empty(
            (N_DATA, class_output_size), dtype=np.int32)
        self.train_Y_chars = np.empty(
            (N_DATA, char_output_size), dtype=np.int32)
        for i in range(N_DATA):
            text = int_encoded[i:i + seuqnce_length]
            data[i] = np.array(text).reshape((seuqnce_length, 1))
            char_int = self.int2char[int_encoded[i + seuqnce_length]]

            class_val = self.char2tup[char_int][0] // 10
            self.train_Y_classes[i] = one_hot_encode(
                class_val, self.train_Y_classes.shape[1])

            char_val = self.char2tup[char_int][1]
            self.train_Y_chars[i] = one_hot_encode(
                char_val, self.train_Y_chars.shape[1])

        self.train_X = data / 350
        self.data_dims = (self.train_X.shape,
                          self.train_Y_classes.shape,
                          self.train_Y_chars.shape
                          )

    def gen_input_from_file(self):
        
        if self.current_read_chars >= self.total_num_chars:
            self.current_read_chars = 0
            self.data_file.seek(0, 0)
            self.raw_datatset = ''
        data = self.data_file.read(self.seuqnce_length * self.batch_size)
        print(self.data_file.tell())
        total_read = len(data)
        batch_size = total_read // self.seuqnce_length
        total_to_read = batch_size * self.seuqnce_length
        data = data[0:total_read]
        if len(data) < self.seuqnce_length:
            self.current_read_chars
            self.data_file.seek(0, 0)
            data = self.data_file.read(self.seuqnce_length * self.batch_size)
            self.raw_datatset = ''
        if len(self.raw_datatset) < self.seuqnce_length:
            self.raw_datatset = data
        else:
            back_index = len(self.raw_datatset) - self.seuqnce_length
            self.raw_datatset = self.raw_datatset[back_index:] + data
        print(self.raw_datatset)
        self.prepare_training_data()
        self.current_read_chars + len(data)
        return self.train_X, self.train_Y_classes, self.train_Y_chars

    def gen_input(self):
        for x, y, z in zip(self.train_X, self.train_Y_classes, self.train_Y_chars):
            yield x, y, z

    def get_batch(self):
        X = []
        Y = []
        Z = []
        if len(self.train_X) // self.batch_size == self.batch:
            self.iterator = self.gen_input()
            self.batch = 0
        for i in range(self.batch_size):
            x, y, z = next(self.iterator)
            X.append(x)
            Y.append(y)
            Z.append(z)
        X = np.stack(X)
        Y = np.stack(Y)
        Z = np.stack(Z)
        self.batch += 1
        return X, Y, Z

    def process(self):
        self.geez_to_dict()
        # self.read_dataset()
        # self.chars_to_dict()
        # self.prepare_training_data()
        # self.iterator = self.gen_input()


# gen = DataGen("data/small.txt", "data/geez.txt", 10, 100)
# gen.process()

# bx, by, bz = gen.get_batch()
# print(by.shape, by.dtype)

# for i in range(10000):
#     batch = gen.gen_input_from_file()
    # print(batch[0].shape)
