from tfg.genetic.CNNGenome import CNNGenome
from tfg.genetic.gene.Gene import Gene
import random


class Mutator:
    """ """

    def mutate(self, genome:CNNGenome):
        original_genes = genome.get_genes()
        mutate_genes = list(map(lambda original_gene: self.random_mutation(original_gene), original_genes))
        return CNNGenome(mutate_genes)

    def random_mutation(self, gene: Gene):
        decision = random.uniform(0, 1)
        return gene.mutate() if decision > 0.5 else gene