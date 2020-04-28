import random
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop

from tfg.genetic.gene.Gene import Gene


class FiltersRangeMutator:
    def __init__(self, min, max, step = 8):
        self.__min = min
        self.__max = max
        self.__step = step

    def mutate(self, input_value):
        return input_value + self.__step if random.uniform(0,1) > 0.5 else input_value - self.__step


class SequentialModelGene(Gene):

    def __init__(self, seed=None, model=None):
        super().__init__()
        self.__seed = seed
        if model is None:
            model = Sequential()

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

        model.summary()

        self.__model = model

    def mutate(self):
        filtersMutator = FiltersRangeMutator(8, 128)
        config = self.__model.get_config()
        for layer in config['layers']:
            if random.uniform(0,1) > 0.3 and layer['class_name'] == 'Conv2D':
                layer_config = layer['config']
                layer_config['filters'] = filtersMutator.mutate(layer_config['filters'])
        mutated_model = Sequential.from_config(config)
        print("SequentialModelGene has been mutated")
        self.__model = mutated_model
        return SequentialModelGene(self.__seed, mutated_model) # TODO or return self

    def build(self):
        # Should be intitialised based on __genome
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.__model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return self.__model
