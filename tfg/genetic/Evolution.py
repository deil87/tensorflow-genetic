from tfg.genetic.Population import Population
from tfg.genetic.CNNIndividual import CNNIndividual
from tfg.genetic.gene.AugmentationGene import AugmentationGene
from tfg.genetic.gene.OptimizerGene import OptimizerGene
from tfg.genetic.gene.OutputActivationGene import OutputActivationGene
from tfg.genetic.gene.SequentialGene import SequentialGene

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

    def __init__(self):
        print("Evolution has been created")

    def run(self):
        print("Evolution has been launched")
        population = Population([
            CNNIndividual(
                [AugmentationGene(), SequentialGene(), OptimizerGene(), OutputActivationGene()]
            ),CNNIndividual(
                [AugmentationGene(), SequentialGene(), OptimizerGene(), OutputActivationGene()]
            ),CNNIndividual(
                [AugmentationGene(), SequentialGene(), OptimizerGene(), OutputActivationGene()]
            )
        ])
        return self