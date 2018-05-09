import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import sys
sys.path.append('..')
from util import *

args = parse_training_args()

from data_generator import DataGen
from models import *
from training_handler import TrainingHandler

batch_size = args.batch_size
epoches = args.epoches

cwd = os.getcwd()
h5_file_path = os.path.join(cwd, args.input_file)
gen = DataGen(h5_file_path, batch_size)

n_batchs = gen.total_batchs
n_iterations = int(epoches * n_batchs)
save_on_every = args.save_on
tag_name = args.tag_name
print("Total Iterations: {0} Total Batchs: {1}".format(n_iterations, n_batchs))

x_dims, y_dims, z_dims = gen.train_x.shape, gen.train_y.shape, gen.train_z.shape
input_shape = x_dims[1:]

char_model = get_char_model(input_shape, z_dims[1], lstm_cell=False)

gen.to_generate = "vowel"
gen.curren_batch = 0
model_name = "char_model"
save_model(char_model, model_name, tag_name)
char_trainer = TrainingHandler(gen, char_model, model_name)
char_trainer.train(tag_name, n_iterations, save_on_every, save_model=True)

gen.close()
