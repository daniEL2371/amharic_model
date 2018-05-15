import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_generator import DataGen
from models import *
from training_handler import TrainingHandler
from util import *
from text_preprocessor import TextPreProcessor

batch_size = 100
seq_length = 100
tp = TextPreProcessor("data/charset.txt", 100, 100)

model = load_best_state("whole_char", "128_GRU")

text = open('data/test.txt', encoding='utf-8').read()
gen = []
for i in range(len(text) - seq_length - 1):
    seed = text[i:i+seq_length]
    target = text[seq_length + i]
    x, y = tp.text_vec(seed, target)
    x = x.reshape((1, x.shape[0], x.shape[1]))
    result = model.predict(x)
    m  = result.argmax()
    gen.append(m)

t = ''.join(tp.nums_to_chars(gen))
print(t)
