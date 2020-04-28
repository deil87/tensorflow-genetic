from tfg.genetic.CNNIndividual import CNNIndividual
from tfg.genetic.gene.AugmentationGene import AugmentationGene
from tfg.genetic.gene.SequentialModelGene import SequentialModelGene


# Consider to rename to CNNGenome and build() should return CNNIndividual
class CNNGenome:

    def __init__(self, genes):
        self.__genes = genes

    def get_genes(self):
        return self.__genes

    # Returns model and corresponding dataGenerator
    def build(self, seed):
        # TODO we should be more generic and support different CNNs. Tensorflow, Keras, GluonCV, MXNet etc.
        # Sequential model gene
        sequential_gene: SequentialModelGene = None
        for gene in self.__genes:
            if isinstance(gene, SequentialModelGene):
                sequential_gene = gene
        model = sequential_gene.build()

        # Augmentation gene
        augmentation_gene:AugmentationGene = None
        for gene in self.__genes:
            if isinstance(gene, AugmentationGene):
                augmentation_gene = gene

        datagen = augmentation_gene.build()

        return CNNIndividual(self, model, datagen)