import gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import json
import numpy as np
import gym
from gym import spaces

from net.net import Net
import pandas as pd
import networkx as nx

from rl.parameters import MAX_STEPS

class Env(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super(Env, self).__init__()

        self.data = data

        self.net = Net()
        self.net.init_from_columns(data.columns)

        self.no_nodes = len(data.columns)

        self.current_step = 0
        self.index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}
        self.max_llog = -np.inf
        self.reward_types_counter = self.get_default_counter_for_reward_types()


    def render(self, **kwargs):
        print(json.dumps(list(self.net.graph.edges)))


    def edge_exists(self, source_node, target_node):
        return self.net.has_edge(
            self.index_to_feature[source_node],
            self.index_to_feature[target_node]
        )

    def write_solution(self):
        with open('./rl_logs/solutions/max_llogs.json', 'a') as f:
            f.write(json.dumps(list(self.net.graph.edges)) + '\n')
            f.flush()

    def get_default_counter_for_reward_types(self):
        return {}

    def get_ordered_reward_types(self):
        return {k: v for k, v in sorted(self.reward_types_counter.items(), key=lambda item: item[1], reverse=True)}
