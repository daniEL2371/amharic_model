import h5py
import numpy as np


class DataGen:

    

    def generate_multi(self, file, n_vowels, n_cons, batch_size=100, seq_length=100, batches=-1):
        output_size = n_cons + n_vowels
        file = h5py.File(file, "r")
        X = file["train_x"]
        total_samples = X.shape[0]
        print(X.shape)
        print(total_samples)
        total_read = 0
        current = 0
        batch = 0
        while True:
            batch_x = np.empty((batch_size, seq_length, output_size))
            batch_y = np.empty((batch_size, output_size))
            if total_samples - total_read < batch_size or batch == batches:
                batch = 0
            
            for b in range(batch_size):
                current = current + b
                upto = current + seq_length
                batch_x[b] = X[current:upto]
                batch_y[b] = X[upto]

            batch += 1
            print("batch ", current)
            total_read = total_read + batch_size
            yield batch_x, batch_y

    def close(self):
        self.dataset.close()


# d = DataGen()
# gen = d.generate_multi('data/test.h5', 10, 46,
#                        batch_size=2, seq_length=4, batches=3)
# for i in range(6):
#     x, y = next(gen)
#     print(x)
