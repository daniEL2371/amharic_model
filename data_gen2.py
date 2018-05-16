import numpy as np
from util import *
import os
import h5py


class DataGen2:

    def __init__(self, charset_file, batch_size, seuqnce_length):
        self.charset_file = charset_file
        self.char2int = {}
        self.int2char = {}
        self.char2tup = {}
        self.batch_size = batch_size
        self.seuqnce_length = seuqnce_length
        self.n_consonants = 0
        self.n_vowels = 0
        self.output_size = 0
        self.load_charset()

    def load_charset(self):
        data = open(self.charset_file, encoding='utf-8').readlines()
        char2int = {}
        int2char = {}
        char2tup = {}
        data[-2] = data[-2] + '\n'
        j = 0
        for k in range(len(data)):
            row = data[k][:-1].split(',')
            for i in range(len(row)):
                char2tup[row[i]] = (k, i)
                int2char[j] = row[i]
                char2int[row[i]] = j
                j += 1
        
        self.int2char = int2char
        self.char2int = char2int
        self.char2tup = char2tup
        self.n_consonants = len(data)
        self.n_vowels = 10

    
    def encode_text_to_num(self, text):
        encoded = [self.char2int[c] for c in text]
        encoded = np.array(encoded).reshape((len(encoded), 1))
        return encoded

    def get_consonants(self, char):
        class_code, vowel_code = self.char2tup[char]
        class_hot = one_hot_encode(class_code, self.n_consonants)
        return class_hot
    
    def get_vowels(self, char):
        class_code, vowel_code = self.char2tup[char]
        vowel_hot = one_hot_encode(vowel_code, self.n_vowels)
        return vowel_hot

    def text_vec(self, text, target):
        output_size = len(self.char2int)
        num_encoded = self.encode_text_to_num2(text)
        hots = []
        for num in num_encoded:
            hots.append(one_hot_encode(num, output_size))
        hots = np.stack(hots)
        output = one_hot_encode(self.char2int[target], output_size)

        return hots, output
    
    def nums_to_chars(self, nums):
        return [self.int2char[i] for i in nums]
    
    def generate_consonants_xy(self, corpus, batches=-1):
        batch_size = self.batch_size
        seq_length = self.seuqnce_length
        to_read = batch_size + seq_length
        tex_data_file = open(corpus, mode='r', encoding='utf-8')
        prev_left = tex_data_file.read(seq_length)
        input_size = len(self.char2int) + 1
        batch = 0
        while True:
            new_batch = tex_data_file.read(batch_size)
            if len(new_batch) < batch_size or batch == batches:
                tex_data_file.seek(0, 0)
                batch = 0
                continue
            seq = prev_left + new_batch
            batch_x = np.empty((batch_size, seq_length, input_size))
            batch_y = np.empty((batch_size, self.n_consonants))
            for b in range(batch_size):
                text = seq[b:seq_length + b]
                taregt = seq[seq_length + b]
                num_encoded = self.encode_text_to_num(text)
                hots = []
                for num in num_encoded:
                    hots.append(one_hot_encode(num, input_size))
                hots = np.stack(hots)
                batch_x[b] = hots
                class_hot = self.get_consonants(taregt)
                batch_y[b] = class_hot
            prev_left = seq[batch_size:seq_length + batch_size]
            batch += 1
            yield batch_x, batch_y
    
    def generate_vowels_xy(self, corpus, batches=-1):
        batch_size = self.batch_size
        seq_length = self.seuqnce_length
        to_read = batch_size + seq_length
        tex_data_file = open(corpus, mode='r', encoding='utf-8')
        prev_left = tex_data_file.read(seq_length)
        input_size = len(self.char2int) + 1
        batch = 0
        while True:
            new_batch = tex_data_file.read(batch_size)
            if len(new_batch) < batch_size or batch == batches:
                tex_data_file.seek(0, 0)
                batch = 0
                continue
            seq = prev_left + new_batch
            batch_x = np.empty((batch_size, seq_length, input_size))
            batch_y = np.empty((batch_size, self.n_vowels))
            for b in range(batch_size):
                text = seq[b:seq_length + b]
                taregt = seq[seq_length + b]
                num_encoded = self.encode_text_to_num(text)
                hots = []
                for num in num_encoded:
                    hots.append(one_hot_encode(num, input_size))
                hots = np.stack(hots)
                batch_x[b] = hots
                class_hot = self.get_vowels(taregt)
                batch_y[b] = class_hot
            prev_left = seq[batch_size:seq_length + batch_size]
            batch += 1
            yield batch_x, batch_y

    def generate_xy(self, corpus, batches=-1):
        batch_size = self.batch_size
        seq_length = self.seuqnce_length
        to_read = batch_size + seq_length
        tex_data_file = open(corpus, mode='r', encoding='utf-8')
        prev_left = tex_data_file.read(seq_length)
        input_size = len(self.char2int) + 1
        batch = 0
        while True:
            new_batch = tex_data_file.read(batch_size)
            if len(new_batch) < batch_size or batch == batches:
                tex_data_file.seek(0, 0)
                batch = 0
                continue
            seq = prev_left + new_batch
            batch_x = np.empty((batch_size, seq_length, input_size))
            batch_y = np.empty((batch_size, input_size))
            for b in range(batch_size):
                text = seq[b:seq_length + b]
                taregt = seq[seq_length + b]
                num_encoded = self.encode_text_to_num(text)
                hots = []
                for num in num_encoded:
                    hots.append(one_hot_encode(num, input_size))
                hots = np.stack(hots)
                batch_x[b] = hots
                output = one_hot_encode(self.char2int[taregt], input_size)
                batch_y[b] = output
            prev_left = seq[batch_size:seq_length + batch_size]
            batch += 1
            yield batch_x, batch_y