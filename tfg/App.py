# Here we can run web application and expose REST API
from tfg.genetic.Evolution import Evolution

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))


class App:
    def __init__(self, speed=0):
        self.speed = speed

    if __name__ == '__main__':
        evolution = Evolution()
        evolution.run()