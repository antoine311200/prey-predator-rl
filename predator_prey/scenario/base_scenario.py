from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from predator_prey.agents import BaseAgent, Entity, EntityType
from utils import torus_distance, torus_offset

# Create enum
from enum import Enum

class WorldType(Enum):
    TORUS = "torus"
    RECTANGLE = "rectangle"

@dataclass
class ScenarioConfiguration:
    agents: list[BaseAgent]
    landmarks: list[Entity]
    width: int
    height: int
    observation_space: spaces.Tuple
    action_space: spaces.Tuple

    damping: float = 0.9


def check_collision(entity1: Entity, entity2: Entity, total_width: int = int(1e6), total_height: int = int(1e6), offset: int = 0):
    width1, height1 = entity1.geometry.width, entity1.geometry.height
    width2, height2 = entity2.geometry.width, entity2.geometry.height

    check_x = (
        (entity1.x - width1 // 2) % total_width < (entity2.x + width2 // 2) % total_width + offset
        and (entity1.x + width1 // 2) % total_width > (entity2.x - width2 // 2) % total_width - offset
    )
    check_y = (
        (entity1.y - height1 // 2) % total_height < (entity2.y + height2 // 2) % total_height + offset
        and (entity1.y + height1 // 2) % total_height > (entity2.y - height2 // 2) % total_height - offset
    )

    if check_x and check_y:
        # print(f"Checking collision between {entity1.name} and {entity2.name}")
        return True


class BaseScenario:

    def __init__(
        self,
        agents: list[BaseAgent],
        landmarks: list[Entity],
        width: int,
        height: int,
        observation_space: spaces.Tuple,
        action_space: spaces.Tuple,
        damping: float = 0.9,
    ):
        self.agents = agents
        self.landmarks = landmarks

        self.damping = damping
        # Set instance size
        self.width = width
        self.height = height

        # Init spaces
        self.observation_space = observation_space
        self.action_space = action_space

        self.mode = WorldType.TORUS

    @property
    def entities(self):
        return self.agents# + self.landmarks

    def _distance(self, agent1: BaseAgent, agent2: BaseAgent) -> float:
        if self.mode == WorldType.RECTANGLE:
            return np.sqrt((agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2)
        elif self.mode == WorldType.TORUS:
            return torus_distance(agent1, agent2, self.width, self.height)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _offset(self, agent1: BaseAgent, agent2: BaseAgent) -> np.ndarray:
        if self.mode == WorldType.RECTANGLE:
            return np.array([agent1.x - agent2.x, agent1.y - agent2.y])
        elif self.mode == WorldType.TORUS:
            return torus_offset(agent1, agent2, self.width, self.height)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def step(self):
        # Simply update the position of the agents based without any physics
        for agent in self.agents:
            agent.x += agent.vx * (2 if agent.type == EntityType("predator") else 0)
            agent.y += agent.vy * (2 if agent.type == EntityType("predator") else 0)

            # Torus world
            if self.mode == WorldType.TORUS:
                agent.x %= self.width
                agent.y %= self.height

            for landmark in self.landmarks:
                if agent != landmark and landmark.can_collide:
                    if check_collision(agent, landmark, self.width, self.height, offset=2):
                        agent.x -= agent.vx
                        agent.y -= agent.vy

            agent.vx *= self.damping
            agent.vy *= self.damping

    def reset(self, **kwargs):
        pass

    def render(self):
        pass

    def observe(self, agent: BaseAgent):
        raise NotImplementedError

    def reward(self, agent: BaseAgent):
        return 0

    def done(self, agent: BaseAgent):
        return False

    def info(self, agent: BaseAgent):
        return {}


from predator_prey.render.geometry import Geometry, Shape


class SimplePreyPredatorScenario(BaseScenario):

    def __init__(
        self,
        n_predators: int,
        n_preys: int,
        width: int,
        height: int,
        landmarks: list[Entity] = None,
    ):

        prey_geometry = Geometry(Shape.CIRCLE, color=(0, 0, 255), x=0, y=0, radius=10)
        predator_geometry = Geometry(
            Shape.CIRCLE, color=(255, 0, 0), x=0, y=0, radius=10
        )

        # Create a list of agent preys and predators
        prey_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * (n_preys + n_predators - 1) + 2,),
            dtype=np.float32,
        )
        prey_action_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.preys = [
            BaseAgent(
                f"prey_{i}",
                EntityType("prey"),
                geometry=prey_geometry,
                observation_space=prey_observation_space,
                action_space=prey_action_space,
            )
            for i in range(n_preys)
        ]
        predator_observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2 * (n_preys + n_predators - 1) + 2,),
            dtype=np.float32,
        )
        predator_action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.predators = [
            BaseAgent(
                f"predator_{i}",
                EntityType("predator"),
                geometry=predator_geometry,
                observation_space=predator_observation_space,
                action_space=predator_action_space,
            )
            for i in range(n_predators)
        ]
        # Scenario spaces
        observation_space = spaces.Tuple(
            [agent.observation_space for agent in self.preys + self.predators]
        )
        action_space = spaces.Tuple(
            [agent.action_space for agent in self.preys + self.predators]
        )
        # Setup the scenario configuration
        config = ScenarioConfiguration(
            agents=self.preys + self.predators,
            landmarks=landmarks,
            width=width,
            height=height,
            observation_space=observation_space,
            action_space=action_space,
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
                    if agent != entity and check_collision(agent, entity):#, self.width, self.height, offset=0):
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
                rel_entity_positions.extend(torus_offset(agent, agent_entity, self.width, self.height))

        for landmark in self.landmarks:
            rel_entity_positions.extend(torus_offset(agent, landmark, self.width, self.height))

        # Add agent velocity
        rel_entity_positions.extend([agent.vx, agent.vy])

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
            for predator in self.predators:
                distance = torus_distance(agent, predator, self.width, self.height)
                reward += 0.01 * distance
        else:
            reward = 0
            distance = min(
                [
                    torus_distance(agent, prey, self.width, self.height)
                    for prey in self.preys
                ]
            )
            reward -= 0.01 * distance
            # for pred in self.predators:
            #     alpha = 0.001 if pred != agent else 0.01
            #     reward -= alpha * min(
            #         [
            #             torus_distance(prey, pred, self.width, self.height)
            #             for prey in self.preys
            #         ]
            #     )

        # print(f"Reward for {agent.name}: {reward}")

        return reward

    def done(self, agent: BaseAgent):
        # If the agent is a prey, it is done if it is caught by a predator
        if agent.type == EntityType("prey"):
            for predator in self.predators:
                radius = (agent.geometry.width / 2 + predator.geometry.width / 2) * 1.5
                # print(
                #     [agent.name, int(agent.x), int(agent.y)],
                #     [predator.name, int(predator.x), int(predator.y)],
                #     int(torus_distance(agent, predator, self.width, self.height)),
                #     [int(a) for a in torus_offset(agent, predator, self.width, self.height)]
                # )
                if torus_distance(agent, predator, self.width, self.height) < radius:
                    return True
        return False