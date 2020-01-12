import pandas
from datasets.datasets import get_dataset
from net.net import Net
import numpy as np


def main():
    data = get_dataset()
    net = Net()
    net.initialize_random_structure(data)
    net.compute_score_per_family(data)
    for i in range(100):
        print(i)
        candidates = [
            net.mutate_through_deletion(data),
            net.mutate_through_inversion(data),
            net.mutate_through_addition(data)
        ]
        winner_index = np.argmax(np.array([net.get_score() for net in candidates]))
        print('winner_index',winner_index)
        winner_net = candidates[winner_index]

        if winner_net.get_score() <= net.get_score():
            print('Step regress the net score')
            break

        net = winner_net
    net.plot()


# main()
