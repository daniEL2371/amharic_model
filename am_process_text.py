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
        self.total_num_chars = 9860  # 14908472
        self.current_read_chars = 0
        self.stride = 1

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

        class_output_size = 36
        char_output_size = 7
        self.data_dims = (
            (None, self.seuqnce_length, 1),
            (None, class_output_size),
            (None, char_output_size))

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
        self.data_dims = (
            (None, seuqnce_length, 1),
            (None, class_output_size),
            (None, char_output_size))
        N_DATA = len(self.raw_datatset) // self.stride - seuqnce_length - 1
        data = np.empty((N_DATA, seuqnce_length, 1), dtype=np.float32)
        self.train_Y_classes = np.empty(
            (N_DATA, class_output_size), dtype=np.int32)
        self.train_Y_chars = np.empty(
            (N_DATA, char_output_size), dtype=np.int32)
        for i in range(N_DATA):
            text = int_encoded[i:i + seuqnce_length * self.stride: self.stride]
            data[i] = np.array(text).reshape((seuqnce_length, 1))
            char_int = self.int2char[int_encoded[i +
                                                 seuqnce_length * self.stride]]

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
        seq_len = self.seuqnce_length * self.stride
        data = self.data_file.read(seq_len * self.batch_size)
        total_read = len(data)
        batch_size = total_read // seq_len
        total_to_read = batch_size * seq_len
        data = data[0:total_read]
        if len(data) < seq_len:
            self.current_read_chars
            self.data_file.seek(0, 0)
            data = self.data_file.read(seq_len * self.batch_size)
            self.raw_datatset = ''
        if len(self.raw_datatset) < seq_len:
            self.raw_datatset = data
        else:
            back_index = len(self.raw_datatset) - seq_len
            self.raw_datatset = self.raw_datatset[back_index:] + data
        self.prepare_training_data()
        self.current_read_chars + len(data)
        return self.train_X, self.train_Y_classes, self.train_Y_chars

    def encode_text_to_num(self, text):
        encoded = [self.char2int[c] for c in text]
        encoded = np.array(encoded).reshape((len(encoded), 1))
        return encoded

    def encode_char(self, char):
        class_code, vowel_code = self.char2tup[char]
        class_hot = one_hot_encode(class_code // 10, 36)
        vowel_hot = one_hot_encode(vowel_code, 7)
        return class_hot, vowel_hot

    def text_to_bin(self):
        batch_size = self.batch_size
        seq_length = self.seuqnce_length
        to_read = batch_size + seq_length
        prev_left = self.data_file.read(seq_length)
        x_file = open("train_x_input.bin", "wb")
        y_file = open("train_y_target_class.bin", "wb")
        z_file = open("train_y_target_vowel.bin", "wb")
        n_rows = 0
        n_chars = len(prev_left)
        while True:
            new_batch = self.data_file.read(batch_size)
            seq = prev_left + new_batch
            new_batch_size = len(new_batch)
            batch_x = np.empty((new_batch_size, seq_length, 1))
            batch_y = np.empty((new_batch_size, 36))
            batch_z = np.empty((new_batch_size, 7))
            for b in range(new_batch_size):
                text = seq[b:seq_length + b]
                taregt = seq[seq_length + b]
                num_encoded = self.encode_text_to_num(text)
                batch_x[b] = num_encoded

                class_hot, vowel_hot = self.encode_char(taregt)
                batch_y[b] = class_hot
                batch_z[b] = vowel_hot
        
            batch_x = batch_x / 350
            batch_x.tofile(x_file)
            batch_y.tofile(y_file)
            batch_z.tofile(z_file)
            n_rows += batch_x.shape[0]
            n_chars += new_batch_size
            prev_left = seq[batch_size:seq_length + batch_size]
            if new_batch_size < batch_size:
                break
        z_file.close()
        y_file.close()
        x_file.close()
        print('Total Rows Saved: {} Total Char: {}'.format(n_rows, n_chars))

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

    def read_from_bin(self, filename):
        file = open(filename, "rb")
        for i in range(10):
            data = np.fromfile(file, count=4, dtype=np.float32)
            print(data)
        file.close()

gen = DataGen("data/small.txt", "data/geez.txt", 100, 100)
gen.process()
# gen.text_to_bin()
gen.read_from_bin('train_y_target_vowel.bin')

# bx, by, bz = gen.get_batch()
# print(by.shape, by.dtype)

# for i in range(10000):
#     batch = gen.gen_input_from_file()
# print(batch[0].shape)

# numpy as np
# import random

# alist = []
# c = 1

# for i in range(1000):
#     alist.append(i)
#     if i == (c * 100):
#         np.array(alist).tofile("file.bin")
#         print alist
#         c = c + 1
#         alist[:] = []  # cl
