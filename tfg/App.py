# Here we can run web application and expose REST API
from tfg.genetic.CNNIndividual import CNNIndividual

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

class App:
    def __init__(self, speed=0):
        self.speed = speed

    if __name__ == '__main__':
        individual = CNNIndividual()
        individual.hello()
        individual.build()
        print("I'm an individual!!!!")