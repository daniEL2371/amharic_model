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
print(tp.nums_to_chars([2,6,7,1]))
# model = load_best_state("whole_char", "128_GRU")

# text = open('data/test.txt', encoding='utf-8').read()
# xs = []
# ys = []
# for i in range(len(text) - seq_length - 1):
#     if i == 100:
#         break
#     seed = text[i:i+seq_length]
#     target = text[seq_length + i]
#     x, y = tp.text_vec(seed, target)
#     xs.append(x)
#     ys.append(y)

# x = np.stack(xs)
# y = np.stack(ys)
# result = model.predict(x, batch_size=100)
# maxes = result.argmax(axis=1)
# print(maxes.shape)