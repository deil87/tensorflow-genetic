from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop

from tfg.genetic.gene.Gene import Gene

class SequentialModelGene(Gene):

    def __init__(self, seed=None):
        super().__init__()
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
        # Should be intitialised based on __genome
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        self.__model = model

    def mutate(self):
        print("SequentialGene has been mutated")
        return self

    def build(self):
        return self.__model