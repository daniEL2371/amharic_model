from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
import os
import time
from check_point_manager import *


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
        self.time_taken = 0
        self.checkpoint = CheckpointManager()

    def train(self, training_tag, n_iterations, save_on, save_model=False):
        self.n_iterations = n_iterations
        self.save_weights_on = save_on
        self.model_tag = training_tag
        s = "{0}_{1}".format(self.model_name, self.model_tag)
        self.checkpoint.prepare(s)
        try:
            os.stat('model_weights')
        except:
            os.mkdir('model_weights')
        self.load_state()
        elapsed_time = 0
        start = time.time()
        for i in range(self.current_iter, self.n_iterations):
            x, y = self.data_generator.get_batch()
            cost = self.model.train_on_batch(x, y)
            if i % self.save_weights_on == 0:
                end = time.time()
                elapsed_time = end - start
                self.current_iter = i
                self.save_state(i, cost, elapsed_time)
                start = time.time()

    def save_state(self, i, cost, elapsed_time):

        file_name = "model_weights/{0}-{1}.txt".format(
            self.model_name, self.model_tag)
        dirname = "{0}-{1}".format(self.model_name, self.model_tag)
        try:
            os.stat(file_name[:-4])
        except:
            os.mkdir(file_name[:-4])
        progress = (i + self.save_weights_on) * 100 / self.n_iterations
        self.time_taken += elapsed_time
        r_iter = self.n_iterations - (i + 1)
        r_time = (elapsed_time) * r_iter/self.save_weights_on
        r_time = self.pretty_time(r_time)
        taken = self.pretty_time(self.time_taken)
        progress = "Progress: {0:.3f}% Batch: {1} Cost: {2:.5f} Time: {3:.3f}s Taken: {4} Remaining: {5}".format(
            progress, self.data_generator.curren_batch, cost, elapsed_time, taken, r_time)
        print(progress)

        now = time.strftime("%Y-%m-%d %H:%M:%S")

        state = "{0},{1},{2},{3},{4},{5},{6},{7},{8}".format(self.n_iterations,
                                                             self.current_iter,
                                                             self.save_weights_on,
                                                             self.latest_weight,
                                                             cost,
                                                             self.data_generator.curren_batch,
                                                             self.time_taken,
                                                             now,
                                                             progress)
        self.latest_weight = "model_weights/{3}/{0}-{1}-{2:.5}.h5".format(
            self.model_name, i, cost,  dirname)
        self.model.save_weights(self.latest_weight)
        self.checkpoint.save(state)

    def load_state(self):
        last_state = self.checkpoint.get_last_state()
        if last_state != None:
            print("Loading State: " + last_state[1])
            vals = last_state[1].split(',')
            self.n_iterations = int(vals[0])
            self.current_iter = int(vals[1]) + 1
            self.save_weights_on = int(vals[2])
            self.latest_weight = vals[3]
            self.model.load_weights(self.latest_weight)
            self.data_generator.curren_batch = int(vals[5]) + 1
            self.time_taken = float(vals[6])

    def load_best_weight(self, tag):
        best_row, min_cost, iter = self.checkpoint.get_best_state()
        file_name = "model_weights/{0}-{3}/{0}-{1}-{2:.5}.h5".format(
            self.model_name, iter, cost, tag)
        self.model.load_weights(file_name)

    def pretty_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        days, hrs = divmod(hrs, 24)
        return "{0} days, {1} hrs, {2} mins, {3:.2f}s".format(days, hrs, mins, secs)

    def clear_old_states(self):
        self.clear_old_states()
