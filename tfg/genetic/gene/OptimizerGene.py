from tfg.genetic.gene.Gene import Gene


class OptimizerGene(Gene):

    def mutate(self):
        print("OptimizerGene has been mutated")
        return self