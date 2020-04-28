

class Population:
    """Population...
    """

    def __init__(self, individuals = None):
        print("Population has been created")
        self.__individuals = individuals

    def get_indevidual(self, index):
        return self.__individuals[index]

    def get_individuals(self):
        return self.__individuals

    def __iter__(self):
        ''' Returns the Iterator object '''
        return PopulationIterator(self)


class PopulationIterator:
    """ Iterator class """


    def __init__(self, team):
        # Team object reference
        self._team = team
        # member variable to keep track of current index
        self._index = 0


    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < (len(self._team.__individuals)):
            if self._index < len(self._team.__individuals):  # Check if junior members are fully iterated or not
                result = (self._team.__individuals[self._index], 'junior')
            else:
                result = (self._team.__individuals[self._index - len(self._team._juniorMembers)], 'senior')
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration
