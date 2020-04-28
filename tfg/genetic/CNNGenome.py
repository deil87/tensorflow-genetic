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
from tfg.genetic.CNNIndividual import CNNIndividual
from tfg.genetic.gene.AugmentationGene import AugmentationGene
from tfg.genetic.gene.Gene import Gene
import random

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

# Consider to rename to CNNGenome and build() should return CNNIndividual
class CNNGenome:

    def __init__(self, genome):
        self.__genome = genome

    # Returns model and corresponding dataGenerator
    def build(self, seed):
        # TODO we should be more generic and support different CNNs. Tensorflow, Keras, GluonCV, MXNet etc.
        model = Sequential()
        # model.batch_input_shape = (86, 28, 28, 1)

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=(28, 28, 1)))

        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25, seed=seed))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25, seed=seed))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5, seed=seed))
        model.add(Dense(10, activation="softmax"))
        # Should be intitialised based on __genome
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        # Augmentation gene
        augmentation_gene:AugmentationGene = None
        for gene in self.__genome:
            if isinstance(gene, AugmentationGene):
                augmentation_gene = gene

        datagen = augmentation_gene.build()

        return CNNIndividual(model, datagen)