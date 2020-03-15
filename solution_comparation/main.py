import json

from ga.mapper import as_net
from ga.parameters import DATA, POPULATION_SIZE
from ga.population import Population
import numpy as np
import matplotlib.pyplot as plt
import math
from solution_comparation.cm_plot import make_confusion_matrix


def main():
    n = 8
    with open('solution_as_edges.json') as file:
        solutions = json.load(file)

    original = set([f'{i[0]}{i[1]}' for i in solutions['original']])
    predicted = set([f'{i[0]}{i[1]}' for i in solutions['predicted']])

    cm = {
        'tn': 0,
        'tp': len(original.intersection(predicted)),
        'fp': len(predicted - original),
        'fn': len(original - predicted)
    }

    cm['tn'] = n * (n - 1) / 2 - cm['tp'] - cm['fp'] - cm['fn']

    labels = ['True Neg', 'FalsePos', 'FalseNeg', 'TruePos']
    categories = ['Zero', 'One']
    make_confusion_matrix(np.array([[cm['tn'],cm['fp']],[cm['fn'],cm['tp']]]),
                          group_names=labels,
                          categories=categories,
                          figsize=(4,4),
                          sum_stats=False,
                          )
    print(cm)
    print_metrics(cm)
    # plt.show()


def print_metrics(cm):
    metrics = {}
    metrics['Accuracy'] = (cm['tp'] + cm['tn']) / (sum(cm.values()))
    metrics['Precision'] = (cm['tp']) / (cm['tp'] + cm['fp'])
    metrics['Recall'] = (cm['tp']) / (cm['tp'] + cm['fn'])
    metrics['Matthews Correlation Coefficient'] = (cm['tp']*cm['tn'] - cm['fp']*cm['fn'])/math.sqrt((cm['tp']+cm['fp'])*(cm['tp']+cm['fn'])*(cm['tn']+cm['fp'])*(cm['tn']+cm['fn']))
    for name,val in metrics.items():
        print(f'{name}: {round(val,4)}')


main()

def plot_net():
    node_ordering = node_ordering = {0: 'asia', 1: 'tub', 2: 'smoke', 3: 'bronc', 4: 'lung', 5: 'either', 6: 'dysp',
                                     7: 'xray'}
    Population.init_state(POPULATION_SIZE, DATA.columns, node_ordering)

    new = as_net(np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]))
    new.plot()
# plot_net()

