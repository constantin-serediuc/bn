import pandas
from datasets.datasets import get_dataset
from net.net import Net
import numpy as np
import json
import networkx as nx
import time

from score import mi

import matplotlib.pyplot as plt
import random


def main():
    data = get_dataset()
    G = nx.Graph()
    nodes = data.columns
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            G.add_edge(nodes[i], nodes[j], weight=mi(nodes[i], nodes[j], data))

    T = nx.maximum_spanning_tree(G)
    terminal_nodes = [x for x in T.nodes() if T.degree(x) == 1]
    root = random.choice(terminal_nodes)
    print(root)
    T = nx.bfs_tree(T, root)
    # net = Net()
    # net.graph = T
    # net.compute_score_per_family(data)
    # print(net.get_score())
    # print(json.dumps(list(net.graph.edges)))

main()
# execution_times = []
# for i in range(100):
#     t = time.time()
#     main()
#     execution_times.append(time.time()-t)
# results = np.array(execution_times)
# print(np.mean(results))
# print(np.std(results))