class Fitness:
    """...

        """

    def __init__(self, metrics_as_map):
        self.__metrics = metrics_as_map

    def get_valid_loss(self):
        return self.__metrics['val_loss']

    def get_valid_accuracy(self):
        return self.__metrics['val_accuracy']