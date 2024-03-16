from dataclasses import dataclass

import numpy as np

from predator_prey.agents import BaseAgent, Entity, EntityType


@dataclass
class ScenarioConfiguration:
    agents: list[BaseAgent]
    landmarks: list[Entity]
    communication_channels: int
    width: int
    height: int

    damping: float = 0.9


def check_collision(entity1: Entity, entity2: Entity):
    width1, height1 = entity1.geometry.width, entity1.geometry.height
    width2, height2 = entity2.geometry.width, entity2.geometry.height

    check_x = (
        entity1.x - width1 // 2 < entity2.x + width2 // 2
        and entity1.x + width1 // 2 > entity2.x - width2 // 2
    )
    check_y = (
        entity1.y - height1 // 2 < entity2.y + height2 // 2
        and entity1.y + height1 // 2 > entity2.y - height2 // 2
    )

    if check_x and check_y:
        # print(f"Checking collision between {entity1.name} and {entity2.name}")
        return True


class BaseScenario:

    def __init__(
        self,
        agents: list[BaseAgent],
        landmarks: list[Entity],
        communication_channels: int,
        width: int,
        height: int,
        damping: float = 0.9,
    ):
        self.agents = agents
        self.landmarks = landmarks
        self.communication_channels = communication_channels

        self.damping = damping
        # Set instance size
        self.width = width
        self.height = height

    @property
    def entities(self):
        return self.agents + self.landmarks

    def step(self):
        # Set the physics of the environment
        # Check for collisions to the landmarks and update the position of the agents accordingly
        # applied_forces = [None for _ in self.entities]
        # for entity_1 in self.entities:
        #     force = np.array([0, 0])

        #     if entity_1.can_move and entity_1.can_collide:
        #         for entity_2 in self.entities:
        #             if entity_1 != entity_2 and entity_1.can_collide:
        #                 distance = np.sqrt((entity_2.x - entity_1.x) ** 2 + (entity_2.y - entity_1.y) ** 2)
        #                 force += (entity_2.x - entity_1.x) / distance, (entity_2.y - entity_1.y) / distance

        #     applied_forces.append(force)

        # Simply update the position of the agents based without any physics
        for agent in self.agents:
            agent.x += agent.vx
            agent.y += agent.vy

            for entity in self.entities:
                if agent != entity and entity.can_collide:
                    if check_collision(agent, entity):
                        agent.x -= agent.vx
                        agent.y -= agent.vy

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

    def __init__(
        self,
        n_predators: int,
        n_preys: int,
        width: int,
        height: int,
        landmarks: list[Entity] = None,
        communication_channels: int = 0,
    ):

        prey_geometry = Geometry(Shape.CIRCLE, color=(0, 0, 255), x=0, y=0, radius=10)
        predator_geometry = Geometry(
            Shape.CIRCLE, color=(255, 0, 0), x=0, y=0, radius=10
        )

        # Create a list of agent preys and predators
        preys = [
            BaseAgent(
                f"prey_{i}",
                EntityType("prey"),
                communicate=False,
                geometry=prey_geometry,
            )
            for i in range(n_preys)
        ]
        predators = [
            BaseAgent(
                f"predator_{i}",
                EntityType("predator"),
                communicate=True,
                geometry=predator_geometry,
            )
            for i in range(n_predators)
        ]

        # Setup the scenario configuration
        config = ScenarioConfiguration(
            agents=preys + predators,
            landmarks=landmarks,
            communication_channels=communication_channels,
            width=width,
            height=height,
        )
        # Dataclass to mapping
        config = config.__dict__
        super().__init__(**config)

    def reset(self):
        for i, agent in enumerate(self.agents):
            is_colliding = True
            while is_colliding:
                x = np.random.uniform(self.width * 0.1, self.width * 0.9)
                y = np.random.uniform(self.height * 0.1, self.height * 0.9)
                is_colliding = False
                agent.set_position(x, y)
                for entity in self.entities:
                    if agent != entity and check_collision(agent, entity):
                        is_colliding = True

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

        for landmark in self.landmarks:
            rel_entity_positions.append([landmark.x - agent.x, landmark.y - agent.y])

        return np.array(rel_entity_positions)

    def reward(self, agent: BaseAgent):
        # Compute distance-based reward for the agent
        # The closer the agent is to a prey, the higher the reward
        # The closer the agent is to a predator, the lower the reward

        # In this simple prey-predator scenario, we can use the list of preys and predators to compute the reward easily
        # The goal will simply be to be as far as possible from the predators for the preys
        # and as close as possible to the preys for the predators
        is_prey = agent.type == EntityType("prey")

        if is_prey:
            reward = 0
            for predator in self.agents:
                if predator.type == EntityType("predator"):
                    distance = np.sqrt(
                        (predator.x - agent.x) ** 2 + (predator.y - agent.y) ** 2
                    )
                    reward -= distance
        else:
            reward = 0
            for prey in self.agents:
                if prey.type == EntityType("prey"):
                    distance = np.sqrt(
                        (prey.x - agent.x) ** 2 + (prey.y - agent.y) ** 2
                    )
                    reward += distance

        return reward
