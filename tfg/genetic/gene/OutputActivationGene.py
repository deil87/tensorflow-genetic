from tfg.genetic.gene.Gene import Gene


class OutputActivationGene(Gene):

    def mutate(self):
        print("OutputActivationGene has been mutated")
        return self