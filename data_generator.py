import os
import h5py


class DataGen:

    def __init__(self, file_name, batch_size, seuqnce_length):
        self.dataset = h5py.File(file_name, "r")
        self.batch_size = batch_size
        self.seq_len = seuqnce_length
        self.train_x = self.dataset["train_x"]
        self.train_y = self.dataset["train_y"]
        self.train_z = self.dataset["train_z"]
        self.curren_batch = 0
        self.total_batchs = self.train_x.shape[0]//self.batch_size
        self.to_generate = "class"
        self.iterator = self.generate_batch()

    def read_class_batch(self, batch):
        batchs = self.train_x.shape[0]//self.batch_size
        current = batch * self.batch_size
        upto = current + self.batch_size
        x_batch = self.train_x[current: upto]
        y_batch = self.train_y[current: upto]
        return x_batch, y_batch

    def read_vowel_batch(self, batch):
        batchs = self.train_x.shape[0]//self.batch_size
        current = batch * self.batch_size
        upto = current + self.batch_size
        x_batch = self.train_x[current: upto]
        z_batch = self.train_z[current: upto]
        return x_batch, z_batch

    def generate_batch(self):
        while True:
            if self.curren_batch >= self.total_batchs:
                self.curren_batch = 0
            if self.to_generate == "class":
                yield self.read_class_batch(self.curren_batch)
            else:
                yield self.read_vowel_batch(self.curren_batch)
            self.curren_batch += 1
    
    def get_batch(self):
        x, y = next(self.iterator)
        return x, y


    def close(self):
        self.dataset.close()


# batch_size = 4
# file_path = "data/small.h5"
# dataGen = DataGen(file_path, batch_size, 25)
# dataGen.to_generate = "class"
# x, y = dataGen.get_batch()

# dataGen.close()
