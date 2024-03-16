from gymnasium.envs.registration import register

register(
    id="MAPredatorPrey-v0",
    entry_point="predator_prey.envs:MultiAgentEnvionment",
)
