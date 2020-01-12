import json

from ga.mapper import as_net
from ga.node_ordering import order
from ga.parameters import DATA, POPULATION_SIZE
from ga.population import Population
import numpy as np


def main():
    n = 8
    with open('solution_as_edges.json') as file:
        solutions = json.load(file)

    original = set([f'{i[0]}{i[1]}' for i in solutions['original']])
    predicted = set([f'{i[0]}{i[1]}' for i in solutions['predicted']])

    confusion_matrix = {
        'tn': 0,
        'tp': len(original.intersection(predicted)),
        'fp': len(predicted - original),
        'fn': len(original - predicted)
    }

    confusion_matrix['tn'] = n * (n - 1) / 2 - confusion_matrix['tp'] - confusion_matrix['fp'] - confusion_matrix['fn']
    print(confusion_matrix)


# main()

def plot_net():
    node_ordering = node_ordering = {0: 'asia', 1: 'tub', 2: 'smoke', 3: 'bronc', 4: 'lung', 5: 'either', 6: 'dysp',
                                     7: 'xray'}
    Population.init_state(POPULATION_SIZE, DATA.columns, node_ordering)

    new = as_net(np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]))
    new.plot()
# plot_net()
