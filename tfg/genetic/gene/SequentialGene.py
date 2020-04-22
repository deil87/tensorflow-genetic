from tfg.genetic.gene.Gene import Gene
import tensorflow as tf


class SequentialGene(Gene):

    def __init__(self):
        self.setLayers([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
        ])

    def setLayers(self, layers):
        self.__layers = layers

    def mutate(self):
        print("SequentialGene has been mutated")
        return self