from ga import mapper
import numpy as np


class Population(object):
    shape = ()
    index_to_feature = {}
    len_features = 0

    @staticmethod
    def init_state(size, features):
        Population.shape = (
            size,
            mapper.get_array_representation_len(len(features))
        )
        Population.index_to_feature = {i: feature for (i, feature) in list(zip(range(len(features)), features))}
        len_features = len(features)

    def __init__(self, as_array=[], as_nets=[]):
        self.population_as_array = as_array
        self.population_as_nets = as_nets

    def random(self):
        self.population_as_array = np.random.randint(2, size=Population.shape)
        self.population_as_nets = mapper.as_nets(self.population_as_array)

    def get(self, i):
        return {
            'as_array': self.population_as_array[i],
            'as_net': self.population_as_nets[i]
        }

    def add_individs(self, individs):
        self.population_as_array = np.append(
            self.population_as_array,
            [individ['as_array'] for individ in individs],
            axis=0
        )
        self.population_as_nets = np.append(
            self.population_as_nets,
            [individ['as_net'] for individ in individs],
            axis=0
        )
