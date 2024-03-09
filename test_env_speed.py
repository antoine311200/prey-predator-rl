import time

import gymnasium as gym

if __name__ == "__main__":
    env = gym.make_vec(
        "predator_prey.envs:PredatorPrey-v0",
        render_mode=None,
        num_envs=10,
        vectorization_mode="sync",
    )
    observation = env.reset()
    start = time.perf_counter()
    for _ in range(10_000):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
    env.close()
    print("Time took:", time.perf_counter() - start)
