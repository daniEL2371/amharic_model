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
tag_name = "2_256"
save_on_every = 10

cwd = os.getcwd()
charset = os.path.join(cwd, charset)
corpus = os.path.join(cwd, corpus)
d = DataGen2(charset, batch_size, seuqnce_length)
gen = d.generate_multi(corpus, batches=batches)

input_shape = (seuqnce_length, (d.n_consonants+ d.n_vowels))
output_shapes = [d.n_consonants, d.n_vowels]

model = multi_task(input_shape, output_shapes, lstm=True)

model_name = "multi_model"
save_model(model, model_name, tag_name)
trainer = TrainingHandler(model, model_name)
trainer.train(tag_name, gen, epoches, batches,
              save_on_every, save_model=True)