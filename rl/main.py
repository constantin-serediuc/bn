import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

from ga.parameters import DATA
from rl.net_env import NetEnv

env = NetEnv(DATA)
check_env(env)

obs = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

for step in range(10):
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(1)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break






# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: NetEnv()])
#
# model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./test/") #add tensorboard_log="./test/" and run tensorboard --logdir /Users/constantin/Documents/bn/rl/test/PPO2_1
# model.learn(total_timesteps=10000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
