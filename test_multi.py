import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_gen2 import DataGen2
from models import *
from training_handler import TrainingHandler
from util import *
from text_preprocessor import TextPreProcessor
import keras

batch_size = 100
seq_length = 100
d = DataGen2("data/charset.txt", 100, 100)

model = load_best_state("multi_model", "2_256")
model.summary()
text = open('data/small.txt', encoding='utf-8').read()[:500]
seed = text[0:0 + seq_length]
print(seed)
gen = []
for i in range(100):
    target = text[seq_length + i]
    x = d.encode_text(seed)
    x = x.reshape((1, x.shape[0], x.shape[1]))
    r1, r2 = model.predict(x)
    vec = np.hstack((r1, r2))
    char = d.vec_to_char(vec.flatten())
    seed += char
    seed = seed[1:]
    gen.append(char)

print(''.join(gen))

