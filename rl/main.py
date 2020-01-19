import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

from ga.parameters import DATA
from rl.net_env import NetEnv

env = DummyVecEnv([lambda: NetEnv(DATA)])

model = PPO2(MlpPolicy, env, verbose=1) #add tensorboard_log="./test/" and run tensorboard --logdir /Users/constantin/Documents/bn/rl/test/PPO2_1
model.learn(total_timesteps=9*10**6)

obs = env.reset()
for i in range(3):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
