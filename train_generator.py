import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=128)
parser.add_argument('-e' '--epochs', dest='epoches', type=float, default=1)
parser.add_argument('-s', '--save_on', dest='save_on', default=1, type=int)
parser.add_argument('-i', '--input', dest='input_file')
parser.add_argument('-t', '--tag', dest='tag_name')
args = parser.parse_args()

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
print("Total Iterations: {0} Total Batchs: {1}".format(n_iterations, n_batchs))


x_dims, y_dims, z_dims = gen.train_x.shape, gen.train_y.shape, gen.train_z.shape
input_shape = x_dims[1:]

class_model = get_class_model(input_shape, y_dims[1])
char_model = get_char_model(input_shape, z_dims[1])

tag_name = args.tag_name

gen.to_generate = "class"
gen.curren_batch = 0
th = TrainingHandler(gen, class_model, "class_model")
th.train(tag_name, n_iterations, save_on_every, save_model=True)
th.load_best_weight(tag_name)

gen.to_generate = "vowel"
gen.curren_batch = 0
th = TrainingHandler(gen, char_model, "char_model")
th.train(tag_name, n_iterations, save_on_every, save_model=True)
th.load_best_weight(tag_name)

gen.close()
