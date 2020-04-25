

class DataContext:
    """...

    """

    def __init__(self, train, valid, test):
        print("DataContext has been initialized")
        self.__train = train
        self.__valid = valid
        self.__test = test

    def get_train(self):
        return self.__train

    def get_valid(self):
        return self.__valid

    def get_test(self):
        return self.__test
