import numpy as np
from util import *
import os
import h5py


class TextPreProcessor:

    def __init__(self, charset_file):
        self.charset_file = charset_file
        self.char2int = {}
        self.int2char = {}
        self.char2tup = {}
        self.tup2char = {}
        self.n_consonants = 0
        self.n_vowels = 0
        self.load_charset()

    def load_charset(self):
        data = open(self.charset_file, encoding='utf-8').readlines()
        char2int = {}
        int2char = {}
        char2tup = {}
        tup2char = {}
        data[-2] = data[-2] + '\n'
        j = 0
        for k in range(len(data)):
            row = data[k][:-1].split(',')
            for i in range(len(row)):
                char2tup[row[i]] = (k, i)
                int2char[j] = row[i]
                char2int[row[i]] = j
                tup = "{0}-{1}".format(k, i)
                tup2char[tup] = row[i]
                j += 1

        self.int2char = int2char
        self.tup2char = tup2char
        self.char2int = char2int
        self.char2tup = char2tup
        self.n_consonants = len(data)
        self.n_vowels = 10

    def encode_char(self, char):
        class_code, vowel_code = self.char2tup[char]
        length = self.n_consonants + self.n_vowels
        x = np.zeros((length,))
        x[class_code] = 1
        x[self.n_consonants + vowel_code] = 1
        return x

    def text_to_bin(self, corpus, filename, batch_size, batches=-1):
        if os.path.exists(filename):
            print("v1 file alread exists")
            return
        input_size = self.n_consonants + self.n_vowels
        output_size = self.n_consonants + self.n_vowels
        chunk_size = input_size * 1024

        tex_data_file = open(corpus, mode='r', encoding='utf-8')
        data_file = h5py.File(filename, "a")

        train_x = data_file.create_dataset('train_x', (0, input_size),
                                           maxshape=(None, input_size),
                                           chunks=(chunk_size, input_size))
        batch = 0
        while True:
            seq = tex_data_file.read(batch_size)
            if len(seq) == 0:
                break
            for b in range(len(seq)):
                c = seq[b]
                x = self.encode_char(c)
                train_x.resize(train_x.shape[0] + 1, axis=0)
                train_x[-1:] = x

        data_file.close()
        tex_data_file.close()

    def vec_to_char(self, vec):
        ic = np.argmax(vec[:self.n_consonants])
        iv = np.argmax(vec[self.n_consonants:])
        key = "{0}-{1}".format(ic, iv)
        c = self.tup2char[key]
        return c

    def generate_multi(self, file, batch_size=100, seq_length=100, batches=-1):
        n_vowels, n_cons = self.n_vowels, self.n_consonants
        output_size = n_cons + n_vowels
        file = h5py.File(file, "r")
        X = file["train_x"]
        total_samples = X.shape[0]
        total_read = 0
        current = 0
        batch = 0
        while True:
            batch_x = np.empty((batch_size, seq_length, output_size))
            batch_y = np.empty((batch_size, output_size))
            if total_samples - total_read < batch_size or batch == batches:
                batch = 0

            for b in range(batch_size):
                current = batch + b
                upto = current + seq_length
                batch_x[b] = X[current:upto]
                batch_y[b] = X[upto]

            batch += 1
            total_read = total_read + batch_size
            batch_y_c = batch_y[:, :n_cons]
            batch_y_v = batch_y[:, n_cons:]
            yield batch_x, [batch_y_c, batch_y_v]


# tp = TextPreProcessor('data/charset.txt')
# tp.text_to_bin('data/test.txt', 'data/test.h5', 5, 100)
# gen = tp.generate_multi('data/test.h5', 10, 46,
#                         batch_size=6, seq_length=8, batches=4)
# for i in range(6):
#     x, y = next(gen)
#     for d in range(len(x)):
#         k, o = x[d], [y[0][d], y[1][d]]
#         seq = []
#         _i, _j = o
#         for g in k:
#             c = tp.vec_to_char(g)
#             seq.append(c)
#         mm = np.hstack((_i, _j))
        # print(seq, " ---> ", tp.vec_to_char(mm))
