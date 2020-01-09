import networkx as nx
import matplotlib.pyplot as plt
from score import entropy
import copy
import random

class Net(object):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.score_per_family = {}
        self.n = 0

    def set_score_per_family(self, scores):
        self.score_per_family = scores

    def set_graph(self, g):
        self.graph = g

    def set_n(self, n):
        self.n = n

    def plot(self):
        nx.draw(self.graph, with_labels=True)
        plt.draw()
        plt.show()

    def initialize_random_structure(self, data): # TODO: true random
        self.graph.add_edge('asia', 'lung')
        self.graph.add_edge('smoke', 'tub')
        self.graph.add_edge('smoke', 'bronc')
        self.graph.add_edge('tub', 'either')
        self.graph.add_edge('lung', 'bronc')
        self.graph.add_edge('either', 'lung')
        self.graph.add_edge('xray', 'either')
        self.graph.add_edge('dysp', 'lung')

    def get_families(self):
        families = {}
        for node in self.graph.nodes:
            families[node] = list(self.graph.predecessors(node))

        return families

    def compute_score_per_family(self, data):
        self.n = data.shape[0]
        for node, parents in self.get_families().items():
            if node in self.score_per_family.keys():
                continue
            self.score_per_family[node] = entropy(node, parents, data)

    def mutate_through_deletion(self, data):
        if len(list(self.graph.edges)) == 0:
            return self

        new_graph = copy.deepcopy(self.graph)
        chosen_edge = random.choice(list(new_graph.edges))
        new_graph.remove_edge(*chosen_edge)

        net = Net()
        net.set_graph(new_graph)
        net.set_n(self.n)

        scores = copy.deepcopy(self.score_per_family)
        scores[chosen_edge[1]] = entropy(chosen_edge[1], list(net.graph.predecessors(chosen_edge[1])), data)
        net.set_score_per_family(scores)

        return net

    def mutate_through_inversion(self, data):
        if len(list(self.graph.edges)) == 0:
            return self

        new_graph = copy.deepcopy(self.graph)
        chosen_edge = random.choice(list(new_graph.edges))

        new_graph.remove_edge(*chosen_edge)
        new_graph.add_edge(chosen_edge[1], chosen_edge[0])

        net = Net()
        net.set_graph(new_graph)
        net.set_n(self.n)

        scores = copy.deepcopy(self.score_per_family)
        for node in chosen_edge:
            scores[node] = entropy(node, list(net.graph.predecessors(node)), data)
        net.set_score_per_family(scores)

        return net

    def mutate_through_addition(self, data):
        non_edges = [i for i in list(nx.non_edges(self.graph)) if list(reversed(i)) not in self.graph.edges]
        if len(non_edges) == 0:
            return self

        chosen_edge = random.choice(non_edges)
        new_graph = copy.deepcopy(self.graph)
        new_graph.add_edge(*chosen_edge)
        net = Net()
        net.set_graph(new_graph)
        net.set_n(self.n)

        scores = copy.deepcopy(self.score_per_family)
        scores[chosen_edge[1]] = entropy(chosen_edge[1], list(net.graph.predecessors(chosen_edge[1])), data)
        net.set_score_per_family(scores)

        return net

    def get_score(self):
        return -1 * sum(self.score_per_family.values()) # am renuntat la * N pentru ca nu face vreo diferenta la compararea de scoruri