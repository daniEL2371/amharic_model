import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_generator import DataGen
from models import *
from training_handler import TrainingHandler
from util import *
from text_preprocessor import TextPreProcessor
import keras

batch_size = 100
seq_length = 100
tp = TextPreProcessor("data/charset.txt", 100, 100)

class_model = load_best_state("class_model", "class_train_128")
vowel_model = load_best_state("char_model", "char_train_128")
# print(class_model.summary())
# print(vowel_model.summary())
text = open('data/test.txt', encoding='utf-8').read()
seed = text[0:0 + seq_length]
for i in range(len(text) - seq_length - 1):
    target = text[seq_length + i]
    x, y = tp.text_vec(seed, target)
    x = x.reshape((1, x.shape[0], x.shape[1]))
    r1 = class_model.predict(x)
    r2 = vowel_model.predict(x)
    c = r1.argmax()
    v = r2.argmax()
    char = tp.cons_vow_to_char(c, v)
    # v = ''.join(tp.nums_to_chars([ch]))
    seed += char
    seed = seed[1:]
    print(char, end='')
# t = ''.join(tp.nums_to_chars(gen))
# print(t)
