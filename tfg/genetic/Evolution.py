import toolz
from keras_preprocessing.image import ImageDataGenerator

from tfg.DataContext import DataContext
from tfg.Evaluator import Evaluator, EvaluatedIndividual
from tfg.genetic.Mutator import Mutator
from tfg.genetic.Population import Population
from tfg.genetic.CNNGenome import CNNGenome
from tfg.genetic.gene.AugmentationGene import AugmentationGene
from tfg.genetic.gene.OptimizerGene import OptimizerGene
from tfg.genetic.gene.OutputActivationGene import OutputActivationGene
from tfg.genetic.gene.SequentialModelGene import SequentialModelGene
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
                [AugmentationGene(), SequentialModelGene(), OptimizerGene(), OutputActivationGene()]
            ),CNNGenome(
                [AugmentationGene(), SequentialModelGene(), OptimizerGene(), OutputActivationGene()]
            ),CNNGenome(
                [AugmentationGene(), SequentialModelGene(), OptimizerGene(), OutputActivationGene()]
            ),CNNGenome(
                [AugmentationGene(), SequentialModelGene(), OptimizerGene(), OutputActivationGene()]
            ),CNNGenome(
                [AugmentationGene(), SequentialModelGene(), OptimizerGene(), OutputActivationGene()]
            )
        ])

        #Build population
        individuals = list(map(lambda cnn_ind: cnn_ind.build(self.__seed), population.get_individuals()))
        print("Individuals" + str(individuals))

        # Evaluate population
        evaluator = Evaluator()
        ev_individuals = list(map(lambda cnn_ind: evaluator.evaluate(cnn_ind, self.__data_context), individuals))

        # Print evaluation results
        self.__print_ev_individuals(ev_individuals)

        # TODO Select parents for cross-over and mutations
        # sorted_ev_individuals = sorted(ev_individuals, key=lambda ev_individual: ev_individual.get_fitness().get_valid_loss(), reverse=True)
        # self.__print_ev_individuals(sorted_ev_individuals)

        parents = toolz.topk(3, ev_individuals, key=lambda ev_individual: ev_individual.get_fitness().get_valid_loss())
        self.__print_ev_individuals(parents)

        # Crossover or mutate individuals
        mutator = Mutator()
        mutated_parents = list(map(lambda parent: mutator.mutate(parent.get_original_genome()), parents))

        # Evaluate offspring

        # Combine original population and offspring

        # Run survival phase

        return self


    def __print_ev_individuals(self, ev_individuals):
        print("\n")
        for idx, ev_ind in enumerate(ev_individuals):
            ev_ind: EvaluatedIndividual = ev_ind
            print("Evaluated {} individual with values = {}".format(idx, ev_ind.get_fitness().get_valid_loss()))
