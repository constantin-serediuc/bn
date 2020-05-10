import os
import sys
sys.path.insert(0,
                os.path.abspath(__file__).rsplit(os.sep, 2)[0])
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, ACKTR

from ga.parameters import DATA
from rl.env_edge_addition import NetEnv
# from rl.env_traverse_graph import NetEnv
# from rl.env_edge_add_remove import NetEnv
import os

import numpy as np
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

env = NetEnv(DATA)
base_test_file = env.FILE
best_mean_reward, n_steps = -np.inf, 0
log_dir = f"./{env.FILE}/monitors"
model_file =log_dir + f'/best_model.pkl'
os.makedirs(log_dir, exist_ok=True)


def callback(_locals, _globals):
    global n_steps, best_mean_reward
    print('=========', n_steps)
    if (n_steps + 1) % 100 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                "Last Best mean reward: {:.2f} - New mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(model_file)
    n_steps += 1
    # Returning False will stop training early
    return True

env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
if os.path.isfile(model_file):
    model = ACKTR.load(model_file, env=env)
else:
    model = ACKTR(MlpLnLstmPolicy, env, tensorboard_log=f"./test{base_test_file}/",
             verbose=0)  # add tensorboard_log="./test/" and run tensorboard --logdir /Users/constantin/Documents/bn/rl/test/PPO2_1
model.learn(total_timesteps=10 ** 5, callback=callback)

# def evaluate(model, num_steps=1000):
#     obs = env.reset()
#     for i in range(num_steps):
#         # _states are only useful when using LSTM policies
#         action, _states = model.predict(obs)
#
#         obs, reward, done, info = env.step(action)
#         env.render()
#
#
# model = PPO2.load("/home/constantin/Desktop/projects/disertation/rl_logs_1_1-20200120T201830Z-001/rl_logs_1_1/1_best_model399.pkl")
# evaluate(model, 30)
