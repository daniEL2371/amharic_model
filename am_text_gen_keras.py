import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_generator import DataGen
from models import *
from am_process_text import *
from training_handler import *

seq_length = 100
batch_size = 100
hidden_size = 128

cwd = os.getcwd()
h5_file_path = os.path.join(cwd, "data/data.h5")
text_file_path = os.path.join(cwd, "data/big.txt")
geez_file = os.path.join(cwd, "data/geez.txt")
processor = DataProcessor(text_file_path, geez_file, batch_size, seq_length)
gen = DataGen(h5_file_path, batch_size, seq_length)

epoches = 1
n_batchs = gen.train_x.shape[0]//batch_size
n_iterations = epoches * n_batchs
save_on_every = batch_size

n_iterations = 1000
save_on_every = 100

x_dims, y_dims, z_dims = gen.train_x.shape, gen.train_y.shape, gen.train_z.shape
input_shape = x_dims[1:]

class_model = get_class_model(input_shape, y_dims[1])
char_model = get_char_model(input_shape, z_dims[1])

gen.to_generate = "class"
gen.curren_batch = 0
th = TrainingHandler(gen, class_model, "class_model")
th.train("256-double", n_iterations, save_on_every, save_model=True)
# th.load_best_weight("256-double")

gen.to_generate = "vowel"
gen.curren_batch = 0
th = TrainingHandler(gen, char_model, "char_model")
th.train("256-double", n_iterations, save_on_every, save_model=True)
# th.load_best_weight("256-double")

# show_len = 100
# start = 200
# seed_text = "ፖለቲካ ጋዜጠኛ ተመስገን ደሳለኝ ባቀረበው ይግባኝ ላይ ክርክር አደረገጋዜጠኛ ተመስገን ደሳለኝ በጠበቃው በአቶ አምሐ መኮንን አማካይነት በፌዴራል ጠቅላይ ፍርድ"
# gen_seq = []

# seed_text = list(seed_text)
# for sh in range(show_len):

#     seed_int = [processor.char2int[s] for s in seed_text]
#     seedvec = np.array(seed_int, dtype=np.float32).reshape(
#         (1, seq_length, 1)) / 350
#     cs = class_model.predict(seedvec)[0].argmax() * 10
#     cr = char_model.predict(seedvec)[0].argmax()
#     char_int = cs + cr
#     if char_int not in processor.int2char:
#         char_int = 350
#     gen_seq.append(char_int)
#     seed_text.append(processor.int2char[char_int])
#     seed_text = seed_text[1:]

# gen_seq = ''.join([processor.int2char[g] for g in gen_seq])
# print(gen_seq)

# gen.close()
