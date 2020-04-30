

class DataContext:
    """...

    """

    def __init__(self, train, test):
        """Data

                        # Arguments
                            train: Tuple (X_train, Y_train)
                            test: X_test

                        """
        self.__train = train
        self.__test = test

    def get_train(self):
        return self.__train

    def train_nrows(self):
        shape = self.__train[1].shape
        return shape[0]

    def get_test(self):
        return self.__test


class SampleDataContext:

    def __init__(self, train, valid):
        """SampleDataContext

                        # Arguments
                            train: Tuple (X_train, Y_train)
                            valid: Tuple (X_valid, Y_valid)

                        """
        self.__train = train
        self.__valid = valid

    def get_train(self):
        return self.__train

    def get_valid(self):
        return self.__valid


