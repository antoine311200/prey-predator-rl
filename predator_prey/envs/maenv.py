from typing import Dict, Tuple

import numpy as np

from predator_prey.scenario import BaseScenario


class MultiAgentEnvionment:
    def __init__(self, scenario: BaseScenario, n_steps=100):
        self.n_steps = n_steps
        self._step = 0

        self.agents = scenario.agents
        self.landmarks = scenario.landmarks

        self.scenario = scenario
        self.observation_space = scenario.observation_space
        self.action_space = scenario.action_space

    def step(
        self, actions: Tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        self._step += 1
        for agent, action in zip(self.agents, actions):
            agent.step(action)

        # world step to add
        self.scenario.step()

        observations = np.array([self.scenario.observe(agent) for agent in self.agents])
        rewards = np.array([self.scenario.reward(agent) for agent in self.agents])
        dones = np.array(
            [self.scenario.done(agent) for agent in self.agents], dtype=bool
        )
        if self._step >= self.n_steps:
            truncated = True#np.array([True for _ in self.agents], dtype=bool)
        else:
            truncated = False#np.array([False for _ in self.agents], dtype=bool)
        infos = {}

        # TODO: Implement shared reward for cooperative agents

        return observations, rewards, dones, truncated, infos

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self._step = 0
        self.scenario.reset(**kwargs)
        info = {}
        # TODO: Get observations from scenario
        observations = np.array([self.scenario.observe(agent) for agent in self.agents])

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
