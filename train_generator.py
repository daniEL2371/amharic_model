import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_generator import DataGen
from models import *
from text_preprocessor import TextPreProcessor
from training_handler import TrainingHandler

seq_length = 100
batch_size = 128

cwd = os.getcwd()
h5_file_path = os.path.join(cwd, "data/data.h5")
text_file_path = os.path.join(cwd, "data/big.txt")
charset_file = os.path.join(cwd, "data/charset.txt")

processor = TextPreProcessor(
    text_file_path, charset_file, batch_size, seq_length)
processor.text_to_bin(h5_file_path)

gen = DataGen(h5_file_path, batch_size, seq_length)

epoches = 1
n_batchs = gen.train_x.shape[0] // batch_size
n_iterations = epoches * n_batchs
save_on_every = batch_size

n_iterations = 200
save_on_every = 1

x_dims, y_dims, z_dims = gen.train_x.shape, gen.train_y.shape, gen.train_z.shape
input_shape = x_dims[1:]

class_model = get_class_model(input_shape, y_dims[1])
char_model = get_char_model(input_shape, z_dims[1])

tag_name = "256-double"

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
