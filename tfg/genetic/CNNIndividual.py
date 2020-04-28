import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from tfg.DataContext import DataContext
from tfg.genetic.gene.AugmentationGene import AugmentationGene
from tfg.genetic.gene.Gene import Gene


class CNNIndividual:

    def __init__(self, model, datagen):
        self.__model = model
        self.__datagen = datagen

    def get_model(self):
        return self.__model

    def get_datagen(self):
        return self.__datagen