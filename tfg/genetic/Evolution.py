from keras_preprocessing.image import ImageDataGenerator

from tfg.DataContext import DataContext
from tfg.Evaluator import Evaluator, EvaluatedIndividual
from tfg.genetic.Population import Population
from tfg.genetic.CNNGenome import CNNGenome
from tfg.genetic.gene.AugmentationGene import AugmentationGene
from tfg.genetic.gene.OptimizerGene import OptimizerGene
from tfg.genetic.gene.OutputActivationGene import OutputActivationGene
from tfg.genetic.gene.SequentialGene import SequentialGene
from toolz import pipe
import tensorflow as tf
from numpy.random import seed as np_random_seed
import random
import os


class Evolution:
    """This is a main class that is responsible for evolution process. Whole time budget is divided between evolutions and generations.
    TimeBudget(
        Evolution_1([
            Generation_1_1(
            ...
            ),
            Generation_1_2(
            ...
            )
        ]),
        Evolution_2([
            Generation_2_1(
            ...
            ),
            Generation_2_2(
            ...
            )
        ]), ...
     )

    """

    def __init__(self, data_context, seed):
        print("Evolution has been created")
        self.__data_context = data_context
        self.__seed = seed

        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np_random_seed(seed)
        tf.random.set_seed(seed)

    def run(self):
        print("Evolution has been launched")
        population = Population([
            CNNGenome(
                [AugmentationGene(), SequentialGene(), OptimizerGene(), OutputActivationGene()]
            ),CNNGenome(
                [AugmentationGene(), SequentialGene(), OptimizerGene(), OutputActivationGene()]
            ),CNNGenome(
                [AugmentationGene(), SequentialGene(), OptimizerGene(), OutputActivationGene()]
            )
        ])

        #Build population
        individuals = list(map(lambda cnn_ind: cnn_ind.build(self.__seed), population.get_individuals()))
        print("Individuals" + str(individuals))

        # Evaluate population
        evaluator = Evaluator()
        ev_individuals = list(map(lambda cnn_ind: evaluator.evaluate(cnn_ind, self.__data_context), individuals))

        # Print evaluation results
        for idx, ev_ind in enumerate(ev_individuals):
            ev_ind: EvaluatedIndividual = ev_ind
            print("Evaluated {} individual with values = {}".format(idx, ev_ind.get_fitness().get_valid_loss()))
        # TODO Select parents for cross-over and mutations


        return self