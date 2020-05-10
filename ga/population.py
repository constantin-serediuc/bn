import json

from ga import mapper
import numpy as np
from ga.parameters import CHECKPOINT_FILE

class Population(object):
    shape = ()
    index_to_feature = {}
    len_features = 0

    @staticmethod
    def init_state(size, features, node_ordering):
        Population.shape = (
            size,
            mapper.get_array_representation_len(len(features))
        )
        Population.index_to_feature = node_ordering
        len_features = len(features)

    def __init__(self, as_array=None, as_nets=None):
        self.population_as_array = as_array
        self.population_as_nets = as_nets

    def random(self):
        self.population_as_array = np.random.choice([0, 1], size=Population.shape, p=[1./10, 9./10])
        self.population_as_nets = mapper.as_nets(self.population_as_array)

    def get(self, i):
        return {
            'as_array': self.population_as_array[i],
            'as_net': self.population_as_nets[i]
        }

    def add_individs(self, individs):
        if self.population_as_array is None:
            self.population_as_array = [individ['as_array'] for individ in individs]
            self.population_as_nets = [individ['as_net'] for individ in individs]
            return

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

    def load(self):
        a = json.load(open(CHECKPOINT_FILE))
        self.population_as_array = np.array(a)
        self.population_as_nets = mapper.as_nets(self.population_as_array)
