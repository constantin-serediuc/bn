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


def as_nets(population_as_array, index_to_feature):
    population_as_nets = []
    for solution in population_as_array:
        net = Net()
        net.graph.add_nodes_from(index_to_feature.values())
        index_of_edges = np.squeeze(np.argwhere(solution == 1))
        edges_as_indexes = [ij(k, len(index_to_feature)) for k in index_of_edges]
        net.graph.add_edges_from([(index_to_feature[i[0]],index_to_feature[i[1]]) for i in edges_as_indexes])
        population_as_nets.append(net)
        # net.plot()
    return population_as_nets


# as_nets(np.array([[1, 1, 0, 0, 0, 1],[1, 1, 0, 0, 0, 1]]), 4)
