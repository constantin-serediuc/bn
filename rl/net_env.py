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

REMOVE_WITH_GREATER_LLOG = 1
REMOVE_BUT_SMALLER_LLOG = 0
REMOVE_BUT_EDGE_NOT_EXISTS = 0
DRAW_WITH_GREATER_LLOG = 1
DRAW_BUT_SMALLER_LLOG = 0
DRAW_BUT_ALREADY_EXISTS = 0
ONLY_MOVE = 0
SAME_NODE = -1


class NetEnv(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super(NetEnv, self).__init__()

        self.data = data
        self.agent_pos = 0  # index in matrix . i = pos/self.n j = pos%self.n

        self.net = Net()
        self.net.init_from_columns(data.columns)

        self.no_nodes = len(data.columns)
        self.n_actions = 3 * self.no_nodes
        self.matrix_count = (self.no_nodes ** 2) - 1
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=self.no_nodes,
                                            shape=((self.no_nodes ** 2) + 1,), dtype=np.float32)

        self.current_step = 0
        self.index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}
        self.types_of_rewards = self.get_default_couter_for_types_of_reward()
        self.max_llog = -np.inf

    def reset(self):
        self.agent_pos = 0

        self.net = Net()
        self.net.init_from_columns(self.data.columns)

        self.current_step = 0
        with open('types_of_rewards_from_colab.json', 'a') as f:
            f.write(json.dumps(self.get_types_of_rewards()) + '\n')
            f.flush()
        self.types_of_rewards = self.get_default_couter_for_types_of_reward()
        return self.current_state()

    def current_state(self):
        return np.append(nx.to_numpy_array(self.net.graph).flatten(), self.agent_pos)

    def step(self, action: int):
        previous_llog = self.net.compute_and_get_score(self.data)
        self.current_step += 1

        draw_edge = True
        remove_edge = False
        if action >= 2 * self.no_nodes:
            remove_edge = True
            draw_edge = False
            action -= 2 * self.no_nodes
        elif action >= self.no_nodes:
            draw_edge = False
            remove_edge = False
            action -= self.no_nodes

        target_node = (action + self.agent_pos) % self.no_nodes

        if target_node == self.agent_pos:
            self.types_of_rewards['same_node'] += 1
            return self.current_state(), SAME_NODE, self.is_finished(), {}

        if not draw_edge and not remove_edge:
            self.agent_pos = target_node
            self.types_of_rewards['only_move'] += 1
            return self.current_state(), ONLY_MOVE, self.is_finished(), {}

        if draw_edge:
            if self.edge_exists(self.agent_pos, target_node):
                self.agent_pos = target_node
                self.types_of_rewards['draw_but_already_exists'] += 1
                return self.current_state(), DRAW_BUT_ALREADY_EXISTS, self.is_finished(), {}

            if self.edge_exists(target_node, self.agent_pos):
                self.net.graph.remove_edge(
                    self.index_to_feature[target_node],
                    self.index_to_feature[self.agent_pos]
                )

            self.net.graph.add_edge(self.index_to_feature[self.agent_pos], self.index_to_feature[target_node])
            self.agent_pos = target_node
            current_llog = self.net.compute_and_get_score(self.data)
            if current_llog > previous_llog:
                reward = DRAW_WITH_GREATER_LLOG
                self.types_of_rewards['draw_with_greater_llog'] += 1
            else:
                reward = DRAW_BUT_SMALLER_LLOG
                self.types_of_rewards['draw_but_smaller_llog'] += 1
            if current_llog > self.max_llog:
                self.max_llog = current_llog
                self.write_solution()
            return self.current_state(), reward, self.is_finished(), {}

        if not self.edge_exists(self.agent_pos, target_node) and not self.edge_exists(target_node, self.agent_pos):
            self.types_of_rewards['remove_but_edge_not_exists'] += 1
            return self.current_state(), REMOVE_BUT_EDGE_NOT_EXISTS, self.is_finished(), {}

        if self.edge_exists(self.agent_pos, target_node):
            self.net.graph.remove_edge(
                self.index_to_feature[self.agent_pos],
                self.index_to_feature[target_node]
            )
        else:
            self.net.graph.remove_edge(
                self.index_to_feature[target_node],
                self.index_to_feature[self.agent_pos]
            )

        current_llog = self.net.compute_and_get_score(self.data)
        if current_llog > previous_llog:
            reward = REMOVE_WITH_GREATER_LLOG
            self.types_of_rewards['remove_with_greater_llog'] += 1
        else:
            reward = REMOVE_BUT_SMALLER_LLOG
            self.types_of_rewards['remove_but_smaller_llog'] += 1

        self.agent_pos = target_node
        if current_llog > self.max_llog:
            self.max_llog = current_llog
            self.write_solution()
        return self.current_state(), reward, self.is_finished(), {}

    def render(self, **kwargs):
        print(json.dumps(list(self.net.graph.edges)))

    def close(self):
        pass

    def is_finished(self):
        adj_matrix = nx.to_numpy_array(self.net.graph)
        if np.count_nonzero(adj_matrix) >= self.matrix_count - self.no_nodes:
            return True
        if self.current_step > MAX_STEPS:
            return True
        return False

    def edge_exists(self, source_node, target_node):
        return self.net.has_edge(
            self.index_to_feature[source_node],
            self.index_to_feature[target_node]
        )

    def get_types_of_rewards(self):
        return {k: v for k, v in sorted(self.types_of_rewards.items(), key=lambda item: item[1], reverse=True)}

    def get_default_couter_for_types_of_reward(self):
        return {'same_node': 0, 'only_move': 0, 'draw_but_already_exists': 0,
                'draw_with_greater_llog': 0, 'draw_but_smaller_llog': 0,
                'remove_but_edge_not_exists': 0, 'remove_with_greater_llog': 0,
                'remove_but_smaller_llog': 0
                }

    def write_solution(self):
        with open('max_llogs.json', 'a') as f:
            f.write(json.dumps(list(self.net.graph.edges)) + '\n')
            f.flush()
