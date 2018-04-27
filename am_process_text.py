import numpy as np
from util import *


class DataGen:

    def __init__(self, dataset_file, geez_file, batch_size, seuqnce_length):
        self.dataset_filename = dataset_file
        self.geez_chars_filename = geez_file
        self.raw_datatset = None
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
        for k in range(34):
            row = l[k]
            for i in range(len(row)):
                char2int[row[i]] = (k * 10, i)

        for k in range(34, 37):
            row = l[k]
            for i in range(len(row)):
                char2int[row[i]] = (340, 0)
        char2int[' '] = (350, 0)
        char_table = l
        self.geez_chars = char2int

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
        self.train_Y_classes = np.empty((N_DATA, class_output_size))
        self.train_Y_chars = np.empty((N_DATA, char_output_size))
        for i in range(N_DATA):
            text = int_encoded[i:i + seuqnce_length]
            data[i] = np.array(text).reshape((seuqnce_length, 1))

            class_val = self.char2tup[self.int2char[int_encoded[i +
                                                                seuqnce_length]]][0] // 10
            self.train_Y_classes[i] = one_hot_encode(
                class_val, self.train_Y_classes.shape[1])

            char_val = self.char2tup[self.int2char[int_encoded[i + seuqnce_length]]][1]
            self.train_Y_chars[i] = one_hot_encode(
                char_val, self.train_Y_chars.shape[1])

        self.train_X = data / 350

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
        self.read_dataset()
        self.chars_to_dict()
        self.prepare_training_data()
        self.iterator = self.gen_input()


# gen = DataGen("data/small.txt", "data/geez.txt", 120, 100)
# gen.process()

# for i in range(10000):
#     batch = gen.get_batch()
