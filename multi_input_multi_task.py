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
train_batches = 100

charset = "data/charset.txt"
train_corpus = "data/train.txt"
<<<<<<< HEAD
val_corpus = "data/validate.txt"
tag_name = "3_256_LSTM"
=======
tag_name = "2_256"
>>>>>>> 065bb7d1e60e70064ee7a772c8983cca829c04d3

seq_length = 128
save_on_every = 100
epoches = 50

cwd = os.getcwd()
charset = os.path.join(cwd, charset)
train_corpus = os.path.join(cwd, train_corpus)

d = DataGen2(charset, batch_size, seq_length)
gen = d.generate_v2(train_corpus, batches=train_batches)

input_shape = (seq_length, (d.n_consonants+ d.n_vowels))
output_shapes = [d.n_consonants, d.n_vowels]

model = multi_task(input_shape, output_shapes, lstm=True)

model_name = "multi_input_multi_task"
trainer = TrainingHandler(model, model_name)
trainer.train(tag_name, gen, epoches, 
              train_batches, save_on_every,
              save_model=True)
