

class Population:
    """Population...
    """

    def __init__(self, individuals = None):
        print("Population has been created")
        self.__individuals = individuals

    def get_indevidual(self, index):
        return self.__individuals[index]
