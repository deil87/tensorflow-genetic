from tfg.genetic import Gene
import tensorflow as tf


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
        print("SequentialGene has been mutated")
        return self