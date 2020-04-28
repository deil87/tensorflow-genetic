import unittest
import pandas as pd
import pathlib
import os

from keras import Sequential
from keras.initializers import glorot_uniform, Orthogonal
from keras.layers import Dense

# neural network with keras tutorial
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from numpy import loadtxt
import tensorflow as tf
from numpy.random import seed as np_random_seed
import random
import os
from keras import backend as K
import numpy as np
from tensorflow_core.python.ops import gen_random_ops


class KerasReproducibility(unittest.TestCase):


    def test_keras_repr(self):
        seed = 1234
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np_random_seed(seed)

        train_path = pathlib.Path(__file__).parent.parent.parent / 'data' / 'pima-indians-diabetes.data.csv'
        # print(train_path)
        # train = pd.read_csv(train_path)[:3000]
        train = loadtxt(train_path, delimiter=',')

        X = train[:, 0:8]
        y = train[:, 8]

        for attempt in range(1,5):
            tf.random.set_seed(seed)
            network_name="cnn_"+str(attempt)
            model = Sequential(name=network_name)
            model.add(Dense(12, input_dim=8, activation='relu', kernel_initializer='glorot_uniform', seed=seed))
            model.add(Dense(8, activation='relu', seed=seed))
            model.add(Dense(1, activation='sigmoid', seed=seed))
            # dummy_optimizer = DummyOptimiser()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(X, y, epochs=5, batch_size=768, shuffle=False)
            # history = model.fit(X, y, epochs=5, batch_size=100, shuffle=False)
            # history = model.fit(X, y, epochs=5, batch_size=10,steps_per_epoch=300, shuffle=False)
            accuracy = history.history['accuracy'][-1]
            print("Attempt {} ,accuracy: {}".format(attempt, accuracy))


if __name__ == '__main__':
    unittest.main()