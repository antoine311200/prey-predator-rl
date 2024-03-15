from dataclasses import dataclass

import numpy as np

from predator_prey.agents import BaseAgent, AgentType

@dataclass
class ScenarioConfiguration:
    agents: list[BaseAgent]
    landmark: list
    communication_channels: int

class BaseScenario:

    def __init__(self, agents: list[BaseAgent], landmark: list, communication_channels: int):
        self.agents = agents
        self.landmark = landmark
        self.communication_channels = communication_channels

    def step(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass


class SimplePreyPredatorScenario(BaseScenario):

    def __init__(self, n_predators: int, n_prey: int, communication_channels: int):
        # Create a list of agent preys and predators
        preys = [BaseAgent(f'prey_{i}', AgentType('prey'), communicate=False) for i in range(n_prey)]
        predators = [BaseAgent(f'predator_{i}', AgentType('predator'), communicate=True) for i in range(n_predators)]

        # Setup the scenario configuration
        config = ScenarioConfiguration(
            agents=preys + predators,
            landmark=[],
            communication_channels=0
        )
        super().__init__(**config)

    def step(self):
        pass

    def reset(self):
        for agent in self.agents:
            agent.x = np.random.uniform(-1, 1)
            agent.y = np.random.uniform(-1, 1)

            agent.vx = 0
            agent.vy = 0

    def render(self):
        pass