from dataclasses import dataclass

import numpy as np

from predator_prey.agents import BaseAgent, AgentType, Entity


@dataclass
class ScenarioConfiguration:
    agents: list[BaseAgent]
    landmark: list[Entity]
    communication_channels: int

    damping: float = 0.9


class BaseScenario:

    def __init__(
        self,
        agents: list[BaseAgent],
        landmark: list[Entity],
        communication_channels: int,
        damping: float = 0.9,
    ):
        self.agents = agents
        self.landmark = landmark
        self.communication_channels = communication_channels

        self.damping = damping

    @property
    def entities(self):
        return self.agents + self.landmark

    def step(self):
        # Set the physics of the environment
        # Check for collisions to the landmarks and update the position of the agents accordingly
        # applied_forces = []
        # for agent in self.agents:
        #     force = np.array([0, 0])

        #     if agent.can_move:
        #         for landmark in self.landmark:
        #             distance = np.sqrt((landmark.x - agent.x) ** 2 + (landmark.y - agent.y) ** 2)
        #             force += (landmark.x - agent.x) / distance, (landmark.y - agent.y) / distance

        #     applied_forces.append(force)

        # Simply update the position of the agents based without any physics
        for agent in self.agents:
            agent.x += agent.vx
            agent.y += agent.vy

            agent.vx *= self.damping
            agent.vy *= self.damping

    def reset(self, **kwargs):
        pass

    def render(self):
        pass

    def observe(self, agent: BaseAgent):
        pass

    def reward(self, agent: BaseAgent):
        pass

    def done(self, agent: BaseAgent):
        pass

    def info(self, agent: BaseAgent):
        pass


from predator_prey.render.geometry import Geometry, Shape

class SimplePreyPredatorScenario(BaseScenario):

    def __init__(self, n_predators: int, n_preys: int, communication_channels: int = 0):

        prey_geometry = Geometry(Shape.CIRCLE, color=(0, 0, 255), x=0, y=0, radius=10)
        predator_geometry = Geometry(Shape.CIRCLE, color=(255, 0, 0), x=0, y=0, radius=10)

        # Create a list of agent preys and predators
        preys = [BaseAgent(f"prey_{i}", AgentType("prey"), communicate=False, geometry=prey_geometry) for i in range(n_preys)]
        predators = [BaseAgent(f"predator_{i}", AgentType("predator"), communicate=True, geometry=predator_geometry) for i in range(n_predators)]

        # Setup the scenario configuration
        config = ScenarioConfiguration(
            agents=preys + predators,
            landmark=[],
            communication_channels=communication_channels
        )
        # Dataclass to mapping
        config = config.__dict__
        super().__init__(**config)

    def step(self):
        pass

    def reset(self, bounds: list[int]):
        for i, agent in enumerate(self.agents):
            x = np.random.uniform(bounds[0]*0.1, bounds[0]*0.9)
            y = np.random.uniform(bounds[1]*0.1, bounds[1]*0.9)
            agent.set_position(x, y)

            agent.vx = 0
            agent.vy = 0

    def render(self):
        pass

    def observe(self, agent: BaseAgent):
        # Observation of all relative positions to the agent of all other entities
        rel_entity_positions = []
        for agent_entity in self.agents:
            if agent_entity != agent:
                rel_entity_positions.append(
                    [agent_entity.x - agent.x, agent_entity.y - agent.y]
                )

        for landmark in self.landmark:
            rel_entity_positions.append([landmark.x - agent.x, landmark.y - agent.y])

        return np.array(rel_entity_positions)

    def reward(self, agent: BaseAgent):
        # Compute distance-based reward for the agent
        # The closer the agent is to a prey, the higher the reward
        # The closer the agent is to a predator, the lower the reward

        # In this simple prey-predator scenario, we can use the list of preys and predators to compute the reward easily
        # The goal will simply be to be as far as possible from the predators for the preys
        # and as close as possible to the preys for the predators
        is_prey = agent.type == AgentType("prey")

        if is_prey:
            reward = 0
            for predator in self.agents:
                if predator.type == AgentType("predator"):
                    distance = np.sqrt(
                        (predator.x - agent.x) ** 2 + (predator.y - agent.y) ** 2
                    )
                    reward -= distance
        else:
            reward = 0
            for prey in self.agents:
                if prey.type == AgentType("prey"):
                    distance = np.sqrt(
                        (prey.x - agent.x) ** 2 + (prey.y - agent.y) ** 2
                    )
                    reward += distance

        return reward
