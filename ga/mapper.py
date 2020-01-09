from ga.population import Population
from net.net import Net
import numpy as np
import math


def ij(k, n):
    i = n - 2 - math.floor(math.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2
    return (int(i), int(j))


def index(i, j, n):
    return int((n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1)


def get_array_representation_len(n):
    return int(n * (n - 1) / 2)


def as_nets(population_as_array):
    population_as_nets = []
    for solution in population_as_array:
        population_as_nets.append(as_net(solution))
    return np.array(population_as_nets)

def as_net(solution):
    net = Net()
    net.graph.add_nodes_from(Population.index_to_feature.values())
    index_of_edges = np.squeeze(np.argwhere(solution == 1))
    edges_as_indexes = [ij(k, len(Population.index_to_feature)) for k in index_of_edges]
    net.graph.add_edges_from([(Population.index_to_feature[i[0]], Population.index_to_feature[i[1]]) for i in edges_as_indexes])
    # net.plot()
    return net

def pass_family_scores_between(new, old_nets):
    for node, parents in new.get_families().items():
        sorted_parents = sorted(parents)

        for net in old_nets:
            if sorted(net.get_families()[node]) == sorted_parents:
                new.score_per_family[node] = net.score_per_family[node]
                continue

    return new
# as_nets(np.array([[1, 1, 0, 0, 0, 1],[1, 1, 0, 0, 0, 1]]), 4)
