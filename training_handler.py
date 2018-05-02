from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os
from timeit import default_timer as timer


class TrainingHandler:

    def __init__(self, data_gen, model, model_name):
        self.n_iterations = 0
        self.current_iter = 0
        self.save_weights_on = 0
        self.data_generator = data_gen
        self.model = model
        self.model_name = model_name
        self.latest_weight = None
        self.model_tag = None
        self.current_batch = 0

    def train(self, training_tag, n_iterations, save_on, save_model=False):
        self.n_iterations = n_iterations
        self.save_weights_on = save_on
        self.model_tag = training_tag
        try:
            os.stat('model_weights')
        except:
            os.mkdir('model_weights')
        self.load_state()
        for i in range(self.current_iter, self.n_iterations):
            start = timer()
            x, y = self.data_generator.get_batch()
            cost = self.model.train_on_batch(x, y)
            elapsed = timer() - start
            if i % self.save_weights_on == 0:
                self.current_iter = i
                self.save_state(i, cost, elapsed)
            

    def save_state(self, i, cost, elapsed_time):

        file_name = "model_weights/{0}-{1}.txt".format(
            self.model_name, self.model_tag)
        dirname = "{0}-{1}".format(self.model_name, self.model_tag)
        try:
            os.stat(file_name[:-4])
        except:
            os.mkdir(file_name[:-4])

        self.latest_weight = "model_weights/{3}/{0}-{1}-{2:.5}.h5".format(
            self.model_name, i, cost,  dirname)
        state = "{0},{1},{2},{3},{4},{5}\n".format(self.n_iterations,
                                                   self.current_iter,
                                                   self.save_weights_on,
                                                   self.latest_weight,
                                                   cost,
                                                   self.data_generator.curren_batch)
        self.model.save_weights(self.latest_weight)
        with open(file_name, mode='a') as file:
            file.write(state)
        progress = (i + self.save_weights_on) * 100 / self.n_iterations
        print("Progress: {0}% Batch: {1} Cost: {2:.5} {3:.3}".format(
            progress, self.data_generator.curren_batch, cost, (elapsed_time)))

    def load_state(self):
        file_name = "model_weights/{0}-{1}.txt".format(
            self.model_name, self.model_tag)
        if os.path.exists(file_name):
            steps = open(file_name).readlines()
            if len(steps) > 0:
                last_step = steps[-1]
                print("Loading State: " + last_step)
                vals = last_step[:-1].split(',')
                self.n_iterations = int(vals[0])
                self.current_iter = int(vals[1]) + 1
                self.save_weights_on = int(vals[2])
                self.latest_weight = vals[3]
                self.model.load_weights(self.latest_weight)
                self.data_generator.curren_batch = int(vals[-1]) + 1

    def load_best_weight(self, tag):
        file_name = "model_weights/{0}-{1}.txt".format(
            self.model_name, tag)
        lines = open(file_name).readlines()
        min_cost_line = lines[0]
        min_cost = 9999999
        iter = 0
        for line in lines:
            vals = line.split(',')
            iter = int(vals[1])
            cost = float(vals[4])
            if cost < min_cost:
                min_cost = cost
                min_cost_line = line
        file_name = "model_weights/{0}-{3}/{0}-{1}-{2:.5}.h5".format(
            self.model_name, iter, cost, tag)
        self.model.load_weights(file_name)
