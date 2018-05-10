import os
import h5py


class DataGen:

    def __init__(self, file_name, batch_size, ):
        self.dataset = h5py.File(file_name, "r")
        self.batch_size = batch_size
        self.train_x = self.dataset["train_x"]
        self.train_y = self.dataset["train_y"]
        try:
            self.train_z = self.dataset["train_z"]
        except:
            pass
        self.current_batch = 0
        self.total_batchs = self.train_x.shape[0] // self.batch_size
        self.to_generate = "class"
        self.iterator = self.generate_batch()

    def read_class_batch(self, batch):
        batchs = self.train_x.shape[0] // self.batch_size
        current = batch * self.batch_size
        upto = current + self.batch_size
        x_batch = self.train_x[current: upto]
        y_batch = self.train_y[current: upto]
        return x_batch, y_batch

    def read_vowel_batch(self, batch):
        batchs = self.train_x.shape[0] // self.batch_size
        current = batch * self.batch_size
        upto = current + self.batch_size
        x_batch = self.train_x[current: upto]
        z_batch = self.train_z[current: upto]
        return x_batch, z_batch

    def generate_batch(self):
        while True:
            if self.current_batch >= self.total_batchs:
                self.current_batch = 0
            if self.to_generate == "class":
                yield self.read_class_batch(self.current_batch)
            else:
                yield self.read_vowel_batch(self.current_batch)
            self.current_batch += 1

    def get_batch(self):
        x, y = next(self.iterator)
        return x, y

    def close(self):
        self.dataset.close()
