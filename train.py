import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("predator_prey.envs:PredatorPreyContinuous-v0", n_envs=5)
model = PPO("MlpPolicy", env, verbose=1)
start_learn = time.perf_counter()
model.learn(total_timesteps=100_000)
print("Time took to learn:", time.perf_counter() - start_learn)


env = make_vec_env("predator_prey.envs:PredatorPreyContinuous-v0", n_envs=1)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")

env.close()
