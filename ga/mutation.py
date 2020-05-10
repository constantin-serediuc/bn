import random
import numpy as np

from ga import mapper
from ga.parameters import MUTATION_RATE
from ga.mapper import as_net, pass_family_scores_between
import operator


def flip_mutate(individ):  # {as_array:[],as_net:Net()}
    new_as_array = np.copy(individ['as_array'])
    for i in range(new_as_array.shape[0]):
        if random.random() < MUTATION_RATE:
            new_as_array[i] = int(not individ['as_array'][i])

    new_net = as_net(new_as_array)
    new_net = pass_family_scores_between(new_net, [individ['as_net']])

    return {'as_array': new_as_array, 'as_net': new_net}


def mutate_worst_gene(individ, index_to_feature):
    feature_to_index = {v: k for k, v in index_to_feature.items()}
    net = individ['as_net']

    in_degrees = dict(net.graph.in_degree(net.graph.nodes))
    max_in_degree_item = max(in_degrees.items(), key=operator.itemgetter(1))
    to = max_in_degree_item[0]

    in_edges = [i for i in list(net.graph.in_edges()) if i[1] == to]
    from_ = random.choice([i[0] for i in in_edges])
    index_to_mutate = mapper.index(feature_to_index[from_], feature_to_index[to], len(index_to_feature.keys()))

    new_as_array = np.copy(individ['as_array'])
    new_as_array[index_to_mutate] = int(not individ['as_array'][index_to_mutate])

    new_net = as_net(new_as_array)
    new_net = pass_family_scores_between(new_net, [individ['as_net']])

    return {'as_array': new_as_array, 'as_net': new_net}


def mutate(individ, index_to_feature, dataset):  # {as_array:[],as_net:Net()}
    individ1 = mutate_worst_gene(individ, index_to_feature)
    individ2 = flip_mutate(individ)

    score_individ1 = individ1['as_net'].compute_and_get_score(dataset)
    score_individ2 = individ2['as_net'].compute_and_get_score(dataset)

    if score_individ1 > score_individ2 :
        return individ1
    return individ2
