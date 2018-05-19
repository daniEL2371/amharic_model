import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import sys
from util import *

from data_generator import DataGen
from data_gen2 import DataGen2
from models import *
from training_handler import TrainingHandler

batch_size = 100
batches = 100
seuqnce_length = 100
epoches = 50
charset = "data/charset.txt"
corpus = "data/big.txt"
tag_name = "char_train"
save_on_every = 10


cwd = os.getcwd()
charset = os.path.join(cwd, charset)
corpus = os.path.join(cwd, corpus)
d = DataGen2(charset, batch_size, seuqnce_length)
gen = d.generate_vowels_xy(corpus, batches=batches)

input_shape = (seuqnce_length, len(d.char2int) + 1)

char_model = get_model(input_shape, d.n_vowels, lstm_cell=True)

model_name = "char_model"
save_model(char_model, model_name, tag_name)
trainer = TrainingHandler(char_model, model_name)
trainer.train(tag_name, gen, epoches, batches,
              save_on_every, save_model=True)