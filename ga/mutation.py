import random
import numpy as np
from ga.parameters import MUTATION_RATE
from ga.mapper import as_net, pass_family_scores_between


def mutate(individ):  # {as_array:[],as_net:Net()}
    new_as_array = np.copy(individ['as_array'])
    for i in range(new_as_array.shape[0]):
        if random.random() < MUTATION_RATE:
            new_as_array[i] = int(not individ['as_array'][i])

    new_net = as_net(new_as_array)
    new_net = pass_family_scores_between(new_net, [individ['as_net']])

    return {'as_array': new_as_array, 'as_net': new_net}
