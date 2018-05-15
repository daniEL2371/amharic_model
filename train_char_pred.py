import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import sys
from util import *

from data_generator import DataGen
from models import *
from training_handler import TrainingHandler

batch_size = 100
epoches = 2
file = "data/data_100_100_v.h5"
tag_name = "first_train"
save_on_every = 10

cwd = os.getcwd()
h5_file_path = os.path.join(cwd, file)
gen = DataGen(h5_file_path, batch_size)

x_dims, y_dims = gen.train_x.shape, gen.train_y.shape
input_shape = (100, x_dims[1], x_dims[2])

char_model = get_model(input_shape, y_dims[1], lstm_cell=False)

gen.curren_batch = 0
model_name = "char_model"
save_model(char_model, model_name, tag_name)
char_trainer = TrainingHandler(gen, char_model, model_name)
char_trainer.train(tag_name, epoches, save_on_every, save_model=True)

gen.close()
