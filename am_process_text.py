import numpy as np
from util import *
import os
import h5py


class DataProcessor:

    def __init__(self, dataset_file, geez_file, batch_size, seuqnce_length):
        self.dataset_filename = dataset_file
        self.geez_chars_filename = geez_file
        self.geez_chars = None
        self.vocab = None
        self.char2int = {}
        self.char2int_norm = {}
        self.char2tup = {}
        self.int2char = {}
        self.batch_size = batch_size
        self.seuqnce_length = seuqnce_length
        self.geez_to_dict()

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
                char2int[row[i]] = (k * 10 + i)
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
        for k in self.char2int.keys():
            self.char2int_norm[k] = self.char2int[k]/350
        class_output_size = 36
        char_output_size = 7

        self.data_dims = (
            (None, self.seuqnce_length, 1),
            (None, class_output_size),
            (None, char_output_size))

    def encode_text_to_num(self, text):
        encoded = [self.char2int[c] for c in text]
        encoded = np.array(encoded).reshape((len(encoded), 1))
        return encoded

    def encode_char(self, char):
        class_code, vowel_code = self.char2tup[char]
        class_hot = one_hot_encode(class_code // 10, 36)
        vowel_hot = one_hot_encode(vowel_code, 7)
        return class_hot, vowel_hot

    def text_to_bin(self, filename):
        batch_size = self.batch_size
        seq_length = self.seuqnce_length
        to_read = batch_size + seq_length
        tex_data_file = open(self.dataset_filename, mode='r', encoding='utf-8')
        prev_left = tex_data_file.read(seq_length)
        data_file = h5py.File(filename, "a")

        train_X = data_file.create_dataset(
            'train_x', (0, seq_length, 1),
            maxshape=(None, seq_length, 1),
            chunks=(1000, seq_length, 1))
        train_y = data_file.create_dataset(
            'train_y', (0, 36),
            maxshape=(None, 36),
            chunks=(1000, 36))
        train_z = data_file.create_dataset(
            'train_z', (0, 7),
            maxshape=(None, 7),
            chunks=(1000, 7))

        n_rows = 0
        n_chars = len(prev_left)
        while True:
            new_batch = tex_data_file.read(batch_size)
            seq = prev_left + new_batch
            if len(new_batch) < batch_size:
                break
            batch_x = np.empty((batch_size, seq_length, 1))
            batch_y = np.empty((batch_size, 36))
            batch_z = np.empty((batch_size, 7))
            for b in range(batch_size):
                text = seq[b:seq_length + b]
                taregt = seq[seq_length + b]
                num_encoded = self.encode_text_to_num(text)
                batch_x[b] = num_encoded

                class_hot, vowel_hot = self.encode_char(taregt)
                batch_y[b] = class_hot
                batch_z[b] = vowel_hot

            batch_x = batch_x / 350
            train_X.resize(train_X.shape[0] + batch_size, axis=0)
            train_X[-batch_size:] = batch_x

            train_y.resize(train_y.shape[0] + batch_size, axis=0)
            train_y[-batch_size:] = batch_y

            train_z.resize(train_z.shape[0] + batch_size, axis=0)
            train_z[-batch_size:] = batch_z

            n_rows += batch_x.shape[0]
            if n_rows % 10000 == 0:
                print(n_rows * 100 / 15000000)
            n_chars += batch_size
            prev_left = seq[batch_size:seq_length + batch_size]
            if batch_size < batch_size:
                break
        data_file.close()
        tex_data_file.close()
        print('Total Rows Saved: {} Total Char: {}'.format(n_rows, n_chars))


# gen = DataGen("data/small.txt", "data/geez.txt", 4, 25)
# gen.text_to_bin("data/small.h5")


