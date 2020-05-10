import copy

from datasets.datasets import get_dataset
from ga.node_ordering import *
from net.net import Net
import numpy as np
import json
import random

def set_family(net, possible_parents, node, dataset):
    initial_score = net.compute_and_get_score(dataset)
    while True:
        scores = []
        for parent_candidate in possible_parents:
            tmp_net = copy.deepcopy(net)
            tmp_net.graph.add_edge(parent_candidate, node)
            scores.append(tmp_net.compute_and_get_score(dataset))

        argmax = np.argmax(np.array(scores))
        if initial_score >= scores[argmax]:
            break
        net.graph.add_edge(possible_parents[argmax], node)
        possible_parents = [i for i in possible_parents if i != possible_parents[argmax]]
        if len(possible_parents) == 0 or len(net.get_families()[node]) >= 3:
            break


def get_k3_stucture(order, dataset):
    net = Net()
    net.init_from_columns(order)
    for i in range(1, len(order)):
        set_family(net, order[:i], order[i], dataset)
    return net

# random_node_ordering = ['asia', 'tub', 'smoke', 'bronc', 'lung', 'either', 'xray', 'dysp']
# random.shuffle(random_node_ordering)
# print(random_node_ordering)

node_order = order_based_on_median(get_dataset())
# print(node_order)
net = get_k3_stucture(node_order, get_dataset())
print(net.compute_and_get_score(get_dataset()))
print(json.dumps(list(net.graph.edges)))

# r = []
# for i in range(2):
#     node_order = order_random(get_dataset())
#     # print(node_order)
#     net = get_k3_stucture(node_order, get_dataset())
#     r.append(net.compute_and_get_score(get_dataset()))
#     # print(net.compute_and_get_score(get_dataset()))
#     # print(json.dumps(list(net.graph.edges)))
# print(r)
# print(sum(r)/2)