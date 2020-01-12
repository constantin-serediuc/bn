from random import random
import numpy as np

from ga import mapper
from ga.mapper import pass_family_scores_between
from ga.parameters import CROSSOVER_RATE


def one_point_crossover(individ1, individ2):
    split_index = int(individ1['as_array'].shape[0] / 2)
    offspring1_array = np.concatenate((individ1['as_array'][:split_index], individ2['as_array'][split_index:]))
    offspring2_array = np.concatenate((individ2['as_array'][:split_index], individ1['as_array'][split_index:]))

    offspring1_net = mapper.as_net(offspring1_array)
    offspring2_net = mapper.as_net(offspring2_array)

    offspring1_net = pass_family_scores_between(offspring1_net, [individ1['as_net'], individ2['as_net']])
    offspring2_net = pass_family_scores_between(offspring2_net, [individ1['as_net'], individ2['as_net']])

    return [{'as_array': offspring1_array, 'as_net': offspring1_net},
            {'as_array': offspring2_array, 'as_net': offspring2_net}]


def crossover(individ1, individ2):  # {as_array:[],as_net:Net()}
    if random() > CROSSOVER_RATE:
        return [individ1, individ2]
    return one_point_crossover(individ1, individ2)
