from tfg.genetic import Gene

class OutputActivationGene(Gene):

    def mutate(self):
        print("OutputActivationGene has been mutated")
        return self