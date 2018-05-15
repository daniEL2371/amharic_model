import h5py


class DataGen:

    def __init__(self, file_name, batch_size, max_batches=-1):
        self.dataset = h5py.File(file_name, "r")
        self.batch_size = batch_size
        self.train_x = self.dataset["train_x"]
        self.train_y = self.dataset["train_y"]
        self.current_batch = 0
        self.total_batches = self.train_x.shape[0] // self.batch_size

    def generate_train_xy(self):
        while True:
            current = self.current_batch * self.batch_size
            upto = current + self.batch_size
            x_batch = self.train_x[current: upto]
            y_batch = self.train_y[current: upto]
            self.current_batch += 1
            if self.current_batch == self.total_batches:
                self.current_batch = 0
            yield x_batch, y_batch

    def close(self):
        self.dataset.close()
