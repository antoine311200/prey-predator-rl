import gymnasium as gym
import numpy as np

from predator_prey.scenario import BaseScenario


class MultiAgentEnvionment(gym.Env):
    def __init__(self, scenario: BaseScenario, n_steps=100):
        self.n_steps = n_steps

        self.agents = scenario.agents
        self.landmarks = scenario.landmarks

        self.scenario = scenario
        self.observation_space = scenario.observation_space
        self.action_space = scenario.action_space

    def step(self, actions):
        for agent, action in zip(self.agents, actions):
            agent.step(action)

        # world step to add
        self.scenario.step()

        observations = []
        rewards = []
        dones = []
        truncated = []
        infos = []
        for agent in self.agents:
            observations.append(self.scenario.observe(agent))
            rewards.append(self.scenario.reward(agent))
            # dones.append(self.scenario.done(agent))
            # infos.append(self.scenario.info(agent))

        # TODO: Implement shared reward for cooperative agents

        return observations, rewards, dones, truncated, infos

    def reset(self, **kwargs):
        self.scenario.reset(**kwargs)
        info = {}
        # TODO: Get observations from scenario
        observations = []
        for agent in self.agents:
            observations.append(self.scenario.observe(agent))

        return observations, info

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __str__(self):
        return "MultiAgentEnvionment"

    def __repr__(self):
        return "MultiAgentEnvionment"
