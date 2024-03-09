import time

import gymnasium as gym

env = gym.make("predator_prey.envs:PredatorPrey-v0", render_mode=None)

observation, info = env.reset(seed=42)
start = time.perf_counter()
total_reward = 0
nb_episodes = 0
for _ in range(100_000):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    total_reward += reward
    if terminated or truncated:
        observation, info = env.reset()
        nb_episodes += 1
print(total_reward, nb_episodes)

env.close()
print("Time took: ", time.perf_counter() - start)
