import numpy as np
import os
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_gen2 import DataGen2
from models import *
from training_handler import TrainingHandler
from util import *
from text_preprocessor import TextPreProcessor
import matplotlib.pyplot as plt

batch_size = 100
batches = 10
seuqnce_length = 100
epoches = 50
charset = "data/charset.txt"
corpus = "data/small.txt"
tag_name = "2_256"
save_on_every = 10

cwd = os.getcwd()
charset = os.path.join(cwd, charset)
corpus = os.path.join(cwd, corpus)
d = DataGen2(charset, batch_size, seuqnce_length)
gen = d.generate_v2(corpus, batches=batches)

model = load_best_state("multi_model", "2_256_s")
print(model.metrics_names)
# loss = model.evaluate_generator(gen, batches)
# print(loss)

# model_wights = get_model_wights("multi_model", "2_256_s")
# m = []
# s = ""
# k = 0
# for w in model_wights:
#     model = keras.models.load_model(w)
#     metric_row = model.evaluate_generator(gen, batches)
#     m.append(metric_row)
#     s += ','.join(str(x) for x in metric_row)
#     s += "\n"
#     k += 1
#     print("{0}/{1}".format(k, len(model_wights)))

# model = keras.models.load_model(model_wights[20])
# metric_row = model.evaluate_generator(gen, batches)
# with open('metric.csv', 'w') as file:
#     file.write(s)

# m = np.array(m)
m = np.loadtxt('metric.csv', dtype=np.float32, delimiter=',')
print(m[:, 0].argmin())
# plt.plot(range(len(m)), m[:, 0], label='loss')
# plt.plot(range(len(m)), m[:, 2], label='dense loss')
# plt.plot(range(len(m)), m[:, 2], label='dense 2 loss')

plt.plot(range(len(m)), m[:, 3])
plt.plot(range(len(m)), m[:, 4])
plt.legend()
plt.show()

# model.summary()
# text = open('data/small.txt', encoding='utf-8').read()[:600]
# seed = text[0:0 + seuqnce_length]
# print(seed)
# gen = []
# for i in range(500):
#     x = d.encode_text(seed)
#     x = x.reshape((1, x.shape[0], x.shape[1]))
#     r1, r2 = model.predict(x)
#     vec = np.hstack((r1, r2))
#     char = d.vec_to_char(vec.flatten())
#     seed += char
#     seed = seed[1:]
#     gen.append(char)

# print(''.join(gen))

