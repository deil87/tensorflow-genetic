from tfg.genetic.Fitness import Fitness
from tfg.genetic.CNNIndividual import CNNIndividual


class Evaluator:
    """...

    """

    def __init__(self):
        print("Evaluator has been instantiated")

    def evaluate(self, individual:CNNIndividual, sample_data_ctx):
        (X_train, Y_train) = sample_data_ctx.get_train()

        # Fit datagen
        datagen = individual.get_datagen()
        datagen.fit(X_train, seed = 1234) # augment = True?

        # Fit model
        model = individual.get_model()

        # TODO should come from corresponding genes
        epochs = 5  # this probably should be a heuristic or be replaced with: min(early stopping, N)
        batch_size = 86
        steps_per_epoch = X_train.shape[0] # batch_size

        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            # We pass some validation for
                            # monitoring validation loss and metrics
                            # at the end of each epoch
                            verbose=2,
                            validation_data=sample_data_ctx.get_valid())

        # history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, seed=2345),
        #                               epochs=epochs, validation_data=data_context.get_valid(),
        #                               verbose=2, steps_per_epoch=steps_per_epoch, shuffle=False # TODO Shuffle = True ???
        #                               , callbacks=[])

        print('Loss training: {} |  Loss valid: {}'.format(history.history['loss'], history.history['val_loss']))
        print('Accuracy training: {} |  accuracy valid: {}'.format(history.history['accuracy'],
                                                                   history.history['val_accuracy']))
        last_val_loss = history.history['val_loss'][-1]
        last_val_accuracy = history.history['val_accuracy'][-1]
        print("Final val loss: {} | final val accuracy: {}".format(last_val_loss, last_val_accuracy) )
        fitness = Fitness({"val_loss": last_val_loss, "val_accuracy": last_val_accuracy})
        return EvaluatedIndividual(individual, fitness)


class EvaluatedIndividual:
    """...

    """

    def __init__(self, individual, fitness: Fitness):
        self.__individual = individual
        self.__fitness = fitness

    def get_individual(self):
        return self.__individual

    def get_original_genome(self):
        individual: CNNIndividual = self.__individual
        return individual.get_genome()

    def get_fitness(self):
        return self.__fitness


