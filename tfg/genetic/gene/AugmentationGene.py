from tfg.genetic import Gene

class AugmentationGene(Gene):

    def mutate(self):
        print("AugmentationGene has been mutated")
        return self