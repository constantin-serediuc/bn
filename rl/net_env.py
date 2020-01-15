import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import gym
from gym import spaces

from net.net import Net
import pandas as pd
import networkx as nx

from rl.parameters import MAX_STEPS

REMOVE_WITH_GREATER_LLOG = 1

REMOVE_BUT_SMALLER_LLOG = 0

REMOVE_BUT_EDGE_NOT_EXISTS = 0

DRAW_WITH_GREATER_LLOG = 1

DRAW_BUT_SMALLER_LLOG = 0

DRAW_BUT_ALREADY_EXISTS = 0

ONLY_MOVE = 0

SAME_NODE = -2


class NetEnv(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super(NetEnv, self).__init__()

        self.data = data
        self.agent_pos = 0  # index in matrix . i = pos/self.n j = pos%self.n

        self.net = Net()
        self.net.init_from_columns(data.columns)

        self.n_actions = 2 * self.net.n
        self.matrix_count = (self.net.n ** 2) - 1

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=self.net.n,
                                            shape=((self.net.n ** 2) + 1,), dtype=np.float32)

        self.current_step = 0
        self.index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}

    def reset(self):
        self.agent_pos = 0

        self.net = Net()
        self.net.init_from_columns(self.data.columns)

        self.current_step = 0

        return self.current_state()

    def current_state(self):
        return np.append(nx.to_numpy_array(self.net.graph).flatten(), self.agent_pos)

    def step(self, action: int):
        previous_llog = self.net.get_score()
        self.current_step += 1

        draw_edge = True
        remove_edge = False
        if action >= 2 * self.net.n:
            remove_edge = True
            action -= 2 * self.net.n
        elif action >= self.net.n:
            draw_edge = False
            action -= self.net.n

        target_node = (action + self.agent_pos) % self.net.n

        if target_node == self.agent_pos:
            return self.current_state(), SAME_NODE, self.is_finished(), {}

        if not draw_edge and not remove_edge:
            self.agent_pos = target_node
            return self.current_state(), ONLY_MOVE, self.is_finished(), {}

        if draw_edge:
            if self.edge_exists(self.agent_pos, target_node):
                self.agent_pos = target_node
                return self.current_state(), DRAW_BUT_ALREADY_EXISTS, self.is_finished(), {}

            if self.edge_exists(target_node, self.agent_pos):
                self.net.graph.remove_edge(
                    self.index_to_feature[target_node],
                    self.index_to_feature[self.agent_pos]
                )

            self.net.graph.add_edge(self.index_to_feature[self.agent_pos], self.index_to_feature[target_node])
            self.agent_pos = target_node
            current_llog = self.net.get_score()
            reward = DRAW_WITH_GREATER_LLOG if current_llog > previous_llog else DRAW_BUT_SMALLER_LLOG
            return self.current_state(), reward, self.is_finished(), {}

        if not self.edge_exists(self.agent_pos, target_node):
            return self.current_state(), REMOVE_BUT_EDGE_NOT_EXISTS, self.is_finished(), {}

        self.net.graph.remove_edge(
            self.index_to_feature[self.agent_pos],
            self.index_to_feature[target_node]
        )
        current_llog = self.net.get_score()
        reward = REMOVE_WITH_GREATER_LLOG if current_llog > previous_llog else REMOVE_BUT_SMALLER_LLOG

        self.agent_pos = target_node

        return self.current_state(), reward, self.is_finished(), {}

    def render(self, **kwargs):
        print(self.net.graph.edges)

    def close(self):
        pass

    def is_finished(self):
        adj_matrix = nx.to_numpy_array(self.net.graph)
        if np.count_nonzero(adj_matrix) >= self.matrix_count - self.net.n:
            return True
        if self.current_step > MAX_STEPS:
            return True
        return False

    def edge_exists(self, source_node, target_node):
        return self.net.has_edge(
            self.index_to_feature[source_node],
            self.index_to_feature[target_node]
        )
