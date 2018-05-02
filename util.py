import numpy as np

def one_hot_encode(val, length):
    a = np.zeros((length,))
    a[val] = 1
    return a