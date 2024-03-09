from gymnasium.envs.registration import register

register(
    id="PredatorPrey-v0",
    entry_point="predator_prey.envs:PredatorPreyEnv",
    max_episode_steps=300,
)


register(
    id="PredatorPreyContinuous-v0",
    entry_point="predator_prey.envs:PredatorPreyContinuousEnv",
    max_episode_steps=300,
)
