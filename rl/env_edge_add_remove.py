import json
import numpy as np
import gym
from gym import spaces

from net.net import Net
import pandas as pd
import networkx as nx

from rl.env import Env
from rl.parameters import MAX_STEPS

# conf1 a2c
# CYCLES = -2
# GREATER_LLOG = 1
# LOWER_LLOG = -1
# EDGE_EXISTS = -1
# EDGE_NOT_EXISTS = -1

CYCLES = -10
GREATER_LLOG = 1
REMOVE_GREATER_LLOG = 2
LOWER_LLOG = -1
EDGE_EXISTS = -1
EDGE_NOT_EXISTS = -1


# cum se modifica reward ul cand is finished
class NetEnv(Env):

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

        self.FILE = 'rl_logs_add_remove'

        self.has_cycles = False
        self.n = self.no_nodes - 1

        self.n_actions = 2 * self.no_nodes * (self.no_nodes - 1)
        self.matrix_count = (self.no_nodes ** 2) - 1

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=self.no_nodes,
                                            shape=((self.no_nodes ** 2),), dtype=np.float32)

    def reset(self):
        self.net = Net()
        self.net.init_graph_nodes(self.data)
        self.current_step = 0
        self.has_cycles = False

        with open(f'./{self.FILE}/reward_types/reward_types.json', 'a') as f:
            f.write(json.dumps(self.get_ordered_reward_types()) + '\n')
            f.flush()

        self.reward_types_counter = self.get_default_counter_for_reward_types()

        return self.current_state()

    def current_state(self):
        return nx.to_numpy_array(self.net.graph).flatten()

    def step(self, action: int):
        previous_llog = self.net.compute_and_get_score(self.data)
        self.current_step += 1

        if action < self.no_nodes * (self.no_nodes - 1):
            return self.process_edge_addition(action, previous_llog)
        else:
            return self.process_edge_elimination(action, previous_llog)

    def render(self, **kwargs):
        print(json.dumps(list(self.net.graph.edges)))

    def is_finished(self):
        if self.has_cycles:
            return True

        adj_matrix = nx.to_numpy_array(self.net.graph)

        if np.count_nonzero(adj_matrix) >= self.matrix_count - self.no_nodes:
            return True
        if self.current_step > MAX_STEPS:
            return True

        return False

    def decode_action(self, action):
        '''
        pt 5 noduri avem matricea
        *   0   1   2   3
        4   *   5   6   7
        8   9   *  10  11
        12 13  14  *   15
        16 17  18 19   *

        self.n = numarul de noduri - 1
        cele de deasupra diagonalei vor avea y+1
        '''
        x = action // self.n
        y = action % self.n
        if y >= x:
            y += 1

        return x, y

    def get_default_counter_for_reward_types(self):
        return {
            'cycles': 0,
            'greater_llog': 0,
            'remove_greater_llog': 0,
            'lower_llog': 0,
            'remove_lower_llog': 0,
            'edge_exists': 0,
            'edge_not_exists': 0
        }

    def process_edge_addition(self, action, previous_llog):
        source, target = self.decode_action(action)
        if self.edge_exists(source, target):
            self.reward_types_counter['edge_exists'] += 1
            return self.current_state(), EDGE_EXISTS, self.is_finished(), {}

        self.net.graph.add_edge(self.index_to_feature[source], self.index_to_feature[target])

        if self.net.has_cycles():
            self.has_cycles = True
            self.reward_types_counter['cycles'] += 1
            return self.current_state(), CYCLES, self.is_finished(), {}

        current_llog = self.net.compute_and_get_score(self.data)
        self.write_bic(current_llog)

        if current_llog <= previous_llog:
            self.reward_types_counter['lower_llog'] += 1
            return self.current_state(), LOWER_LLOG, self.is_finished(), {}

        if current_llog > self.max_llog:
            self.max_llog = current_llog
            self.write_solution(current_llog)
        self.reward_types_counter['greater_llog'] += 1

        return self.current_state(), GREATER_LLOG, self.is_finished(), {}

    def process_edge_elimination(self, action, previous_llog):
        action = action - self.no_nodes * (self.no_nodes - 1)
        source, target = self.decode_action(action)

        if not self.edge_exists(source, target):
            self.reward_types_counter['edge_not_exists'] += 1
            return self.current_state(), EDGE_NOT_EXISTS, self.is_finished(), {}

        self.net.graph.remove_edge(self.index_to_feature[source], self.index_to_feature[target])

        current_llog = self.net.compute_and_get_score(self.data)

        # if self.current_step % 20 == 0:
        self.write_bic(current_llog)

        if current_llog <= previous_llog:
            self.reward_types_counter['remove_lower_llog'] += 1
            return self.current_state(), LOWER_LLOG, self.is_finished(), {}

        if current_llog > self.max_llog:
            self.max_llog = current_llog
            self.write_solution(current_llog)
        self.reward_types_counter['remove_greater_llog'] += 1

        return self.current_state(), REMOVE_GREATER_LLOG, self.is_finished(), {}
