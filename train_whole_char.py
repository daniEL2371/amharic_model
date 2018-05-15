import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import sys
from util import *

from data_generator import DataGen
from models import *
from training_handler import TrainingHandler

batch_size = 100
epoches = 10
file = "data/data_100_500_w.h5"
tag_name = "128_GRU"
save_on_every = 10


cwd = os.getcwd()
h5_file_path = os.path.join(cwd, file)
gen = DataGen(h5_file_path, batch_size)

x_dims, y_dims = gen.train_x.shape, gen.train_y.shape
input_shape = (x_dims[1], x_dims[2])

class_model = get_model(input_shape, y_dims[1], lstm_cell=False)

gen.curren_batch = 0
model_name = "whole_char"
save_model(class_model, model_name, tag_name)
trainer = TrainingHandler(gen, class_model, model_name)
trainer.train(tag_name, epoches, save_on_every, save_model=True)

gen.close()
