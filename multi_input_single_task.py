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
val_batches = 33

charset = "data/charset.txt"
train_corpus = "data/train.txt"
val_corpus = "data/validate.txt"
tag_name = "3_256"

seq_length = 100
save_on_every = 100
epoches = 50

cwd = os.getcwd()
charset = os.path.join(cwd, charset)
train_corpus = os.path.join(cwd, train_corpus)
val_corpus = os.path.join(cwd, val_corpus)

d = DataGen2(charset, batch_size, seq_length)
gen = d.generate_v4(train_corpus, batches=train_batches)
val_gen = d.generate_v4(val_corpus, batches=val_batches)

input_shape = (seq_length, (d.n_consonants+ d.n_vowels))
output_shape = len(d.char2int) + 1

model = get_model(input_shape, output_shape, lstm_cell=True)

model_name = "multi_input_single_task"
trainer = TrainingHandler(model, model_name)
trainer.train(tag_name, gen, epoches, 
              train_batches, save_on_every,
              val_gen=val_gen, val_batches=val_batches, 
              save_model=True)