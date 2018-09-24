from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import LeakyReLU, MaxPooling2D, Flatten
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras import initializers
import keras
import tensorflow as tf
import numpy as np
import l3

l3.anal_word('')

# x = np.random.rand(100, 56)
    

# model = Sequential()
# model.add(Conv2D(5, input_shape=(10, 56, 1),
#                  padding='same', strides=1, 
#                  kernel_size=(10, 56),
#                  activation='linear'))
# model.add(LeakyReLU(alpha=.1))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.summary()

