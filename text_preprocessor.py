import numpy as np
from util import *
import os
import h5py


class TextPreProcessor:

    def __init__(self, dataset_file, charset_file, batch_size, seuqnce_length):
        self.dataset_filename = dataset_file
        self.charset_file = charset_file
        self.char2int = {}
        self.int2char = {}
        self.batch_size = batch_size
        self.seuqnce_length = seuqnce_length
        self.n_consonants = 0
        self.n_vowels = 0
        self.load_charset()

    def load_charset(self):
        data = open(self.charset_file, encoding='utf-8').readlines()
        char2int = {}
        int2char = {}
        char2tup = {}
        data[-2] = data[-2] + '\n'
        for k in range(len(data)):
            row = data[k][:-1].split(',')
            for i in range(len(row)):
                char2int[row[i]] = (k * 10 + i)
                int2char[k * 10 + i] = row[i]
                char2tup[row[i]] = (k, i)

        self.int2char = int2char
        self.char2int = char2int
        self.char2tup = char2tup
        self.n_consonants = len(data)
        self.n_vowels = 10

    def encode_text_to_num(self, text):
        encoded = [self.char2int[c] for c in text]
        encoded = np.array(encoded).reshape((len(encoded), 1))
        return encoded

    def encode_char(self, char):
        class_code, vowel_code = self.char2tup[char]
        class_hot = one_hot_encode(class_code, self.n_consonants)
        vowel_hot = one_hot_encode(vowel_code, self.n_vowels)
        return class_hot, vowel_hot

    def text_to_bin(self, filename):
        if os.path.exists(filename):
            return
        batch_size = self.batch_size
        seq_length = self.seuqnce_length
        to_read = batch_size + seq_length
        tex_data_file = open(self.dataset_filename, mode='r', encoding='utf-8')
        prev_left = tex_data_file.read(seq_length)
        data_file = h5py.File(filename, "a")

        chunk_size = 40 * seq_length * batch_size
        train_X = data_file.create_dataset(
            'train_x', (0, seq_length, 1),
            maxshape=(None, seq_length, 1),
            chunks=(chunk_size, seq_length, 1))
        train_y = data_file.create_dataset(
            'train_y', (0, self.n_consonants),
            maxshape=(None,  self.n_consonants),
            chunks=(chunk_size,  self.n_consonants))
        train_z = data_file.create_dataset(
            'train_z', (0, self.n_vowels),
            maxshape=(None, self.n_vowels),
            chunks=(chunk_size, self.n_vowels))

        n_rows = 0
        n_chars = len(prev_left)
        while True:
            new_batch = tex_data_file.read(batch_size)
            seq = prev_left + new_batch
            if len(new_batch) < batch_size:
                break
            batch_x = np.empty((batch_size, seq_length, 1))
            batch_y = np.empty((batch_size, self.n_consonants))
            batch_z = np.empty((batch_size, self.n_vowels))
            for b in range(batch_size):
                text = seq[b:seq_length + b]
                taregt = seq[seq_length + b]
                num_encoded = self.encode_text_to_num(text)
                batch_x[b] = num_encoded

                class_hot, vowel_hot = self.encode_char(taregt)
                batch_y[b] = class_hot
                batch_z[b] = vowel_hot

            batch_x = batch_x / (self.n_consonants * 10)
            train_X.resize(train_X.shape[0] + batch_size, axis=0)
            train_X[-batch_size:] = batch_x

            train_y.resize(train_y.shape[0] + batch_size, axis=0)
            train_y[-batch_size:] = batch_y

            train_z.resize(train_z.shape[0] + batch_size, axis=0)
            train_z[-batch_size:] = batch_z

            n_rows += batch_x.shape[0]
            if n_rows % 10000 == 0:
                print("{0:.4}%".format(n_rows * 100 / 15000000))
            n_chars += batch_size
            prev_left = seq[batch_size:seq_length + batch_size]
            if batch_size < batch_size:
                break
        data_file.close()
        tex_data_file.close()
        print('Total Rows Saved: {} Total Char: {}'.format(n_rows, n_chars))


