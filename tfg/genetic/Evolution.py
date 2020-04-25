from keras_preprocessing.image import ImageDataGenerator

from tfg.DataContext import DataContext
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

    def __init__(self, data_context):
        print("Evolution has been created")
        self.__data_context = data_context

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
        first_indiv:CNNIndividual = population.get_indevidual(0)
        model = first_indiv.build()

        #TODO following should be also part of selection process. AugmentationGene
        # With data augmentation to prevent overfitting (accuracy 0.99286)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        dc:DataContext = self.__data_context
        (X_train, Y_train) = dc.get_train()
        datagen.fit(X_train)

        # TODO should come from corresponding genes
        epochs = 5  # this probably should be a heuristic or be replaced with: min(early stopping, N)
        batch_size = 86

        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                      epochs=epochs, validation_data=dc.get_valid(),
                                      verbose=2, steps_per_epoch=X_train.shape[0] // batch_size
                                      , callbacks=[])
        print("History\n")
        print('Loss training: {} |  Loss valid: {}'.format(history.history['loss'], history.history['val_loss']))
        print('Accuracy training: {} |  accuracy valid: {}'.format(history.history['accuracy'], history.history['val_accuracy']))
        return self