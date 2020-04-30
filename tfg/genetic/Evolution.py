import copy
import math

import numpy as np
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
import time


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

    def __init__(self, data_context, max_runtime, seed, number_of_evolutions = 3):
        """Configures evolution process for training.

                # Arguments
                    data_context: DataContext instance that provide all the data to train on
                    max_runtime: int maximum number of seconds for this run
                    metrics: List of metrics to be evaluated by the model

                """
        print("Evolution has been created")
        self.__data_context = data_context
        self.__number_of_evolutions = number_of_evolutions
        self.__max_runtime = max_runtime
        self.__seed = seed

        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np_random_seed(seed)
        tf.random.set_seed(seed)

    # credit goes to Georgy Chichladze. Thanks!
    def generate_timeboxes(self, n_boxes, max_runtime):

        def f(coefs, x):
            return coefs * np.exp(x)

        def lengths(coef):
            results = list(map(lambda x: f(coef, x), range(0, n_boxes)))
            return results

        sum_of_exps = sum(list(map(lambda x: np.exp(x), range(0, n_boxes))))
        c = max_runtime / sum_of_exps

        return lengths(c)

    # current_milli_time = lambda: int(round(time.time() * 1000))
    current_seconds_time = lambda self: int(round(time.time()))

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


        timeboxes = self.generate_timeboxes(self.__number_of_evolutions, self.__max_runtime)

        for tb_idx, timebox in enumerate(timeboxes):
            start_tb_time = self.current_seconds_time()

            generation_number = 1
            while self.current_seconds_time() - start_tb_time < timebox:
                #Build population
                individuals = list(map(lambda cnn_ind: cnn_ind.build(self.__seed), population.get_individuals()))
                print("Individuals" + str(individuals))

                # Evaluate population
                evaluator = Evaluator()
                ev_individuals = list(map(lambda cnn_ind: evaluator.evaluate(cnn_ind, self.__data_context), individuals))

                # Print evaluation results
                self.__print_ev_individuals(ev_individuals)

                # TODO Select parents for cross-over and mutations. Note minus sign to pick smallest values
                parents = toolz.topk(3, ev_individuals, key=lambda ev_individual: -ev_individual.get_fitness().get_valid_loss())
                self.__print_ev_individuals(parents)

                # Crossover or mutate individuals
                mutator = Mutator()
                offspring_genomes = list(map(lambda parent: mutator.mutate(parent.get_original_genome()), copy.deepcopy(parents)))

                # Materialize offspring
                offspring = list(map(lambda offspring_genome: offspring_genome.build(self.__seed), offspring_genomes))

                # Evaluate offspring
                ev_offspring = list(map(lambda offspring_ind: evaluator.evaluate(offspring_ind, self.__data_context), offspring))

                print("\n Evaluated offspring \n")
                self.__print_ev_individuals(ev_offspring)

                # Combine original population and offspring
                expanded_individuals = ev_individuals + ev_offspring

                # Run survival phase for evaluated individuals
                survived = toolz.topk(len(individuals), expanded_individuals,
                                     key=lambda ev_individual: -ev_individual.get_fitness().get_valid_loss())

                survived_genomes = list(map(lambda survived_ind: mutator.mutate(survived_ind.get_original_genome()), survived))

                population = Population(individuals=survived_genomes)
                print("Evolution #{} | generation #{} is finished".format(tb_idx, generation_number))
                generation_number += 1

        return self


    def __print_ev_individuals(self, ev_individuals):
        print("\n")
        for idx, ev_ind in enumerate(ev_individuals):
            ev_ind: EvaluatedIndividual = ev_ind
            print("Evaluated {} individual with values = {}".format(idx, ev_ind.get_fitness().get_valid_loss()))
