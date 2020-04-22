from tfg.genetic.gene.Gene import Gene


class AugmentationGene(Gene):

    def mutate(self):
        print("AugmentationGene has been mutated")
        return self