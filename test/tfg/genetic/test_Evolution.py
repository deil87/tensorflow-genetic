import math
import unittest
import pandas as pd
import numpy as np
import pathlib
import os

from tfg.DataContext import DataContext
from tfg.genetic.Evolution import Evolution
import sys
from contextlib import contextmanager
from io import StringIO
import matplotlib.pyplot as plt
from sympy import *
from sympy import Symbol


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class EvolutionTestCase(unittest.TestCase):

    def test_mnist(self):
        # resource_path = os.path.join(os.path.split(__file__)[0], "resources")
        train_path = pathlib.Path(__file__).parent.parent.parent / 'data' / 'mnist' / 'train.csv'
        test_path = pathlib.Path(__file__).parent.parent.parent / 'data' / 'mnist' / 'test.csv'
        # print(train_path)
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        # print(train.head())
        # print(test.head())
        Y_train = train["label"]
        X_train = train.drop(labels=["label"], axis=1)

        #Normalization
        X_train = X_train / 255.0
        test = test / 255.0

        # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
        X_train = X_train.values.reshape(-1, 28, 28, 1)
        test = test.values.reshape(-1, 28, 28, 1)

        # Label encoding
        from keras.utils.np_utils import to_categorical
        Y_train = to_categorical(Y_train, num_classes=10)
        # print(Y_train)

        data_context = DataContext(train = (X_train, Y_train), test=test)
        # print(X_train)
        evolution = Evolution(data_context=data_context, max_runtime= 100, seed=1234, number_of_evolutions=5)

        self.assertTrue(isinstance(evolution, Evolution))

        evolution.run()


if __name__ == '__main__':
    unittest.main()
