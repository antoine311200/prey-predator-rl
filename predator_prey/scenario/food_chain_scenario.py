from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from gymnasium import spaces

from predator_prey.agents import BaseAgent, Entity, EntityType
from predator_prey.render.geometry import Geometry, Shape
from predator_prey.scenario.base_scenario import BaseScenario, WorldType, check_collision
from predator_prey.utils import torus_distance, torus_offset


@dataclass
class IFoodChainAgent:
    type: EntityType
    preys: list[EntityType]
    predators: list[EntityType]
    color: tuple[int, int, int]
    speed: float
    size: int
    targets: list[EntityType] = field(default_factory=list)


SIMPLE_FOODCHAIN_RELATIONS: dict[EntityType, IFoodChainAgent] = {
    EntityType("target"): IFoodChainAgent(
        type=EntityType("target"),
        preys=[],
        predators=[EntityType("low_agent")],
        color=(0, 0, 0),
        speed=0,
        size=5,
    ),
    EntityType("low_agent"): IFoodChainAgent(
        type=EntityType("low_agent"),
        preys=[EntityType("target")],
        predators=[EntityType("high_agent"), EntityType("mid_agent")],
        color=(0, 0, 255),
        speed=5,
        size=20,
    ),
    EntityType("mid_agent"): IFoodChainAgent(
        type=EntityType("mid_agent"),
        preys=[EntityType("low_agent")],
        predators=[EntityType("high_agent"), EntityType("super_agent")],
        color=(0, 255, 0),
        speed=3,
        size=12
    ),
    EntityType("high_agent"): IFoodChainAgent(
        type=EntityType("high_agent"),
        preys=[EntityType("low_agent"), EntityType("mid_agent")],
        predators=[],
        color=(255, 0, 0),
        speed=12,
        size=10
    ),
    EntityType("super_agent"): IFoodChainAgent(
        type=EntityType("super_agent"),
        preys=[EntityType("mid_agent")],
        predators=[],
        color=(255, 255, 0),
        speed=20,
        size=15
    )
}

SIMPLE_LANDMARKS = [
    Entity(
        'target_1',
        EntityType('target'),
        x=300, y=200,
        geometry=Geometry(Shape.CIRCLE, color=(255, 255, 0), x=0, y=0, radius=5),
    ),
    Entity(
        'target_2',
        EntityType('target'),
        x=200, y=200,
        geometry=Geometry(Shape.CIRCLE, color=(255, 255, 0), x=0, y=0, radius=5),
    )
]

class FoodChainScenario(BaseScenario):

    def __init__(
        self,
        food_chain: dict[EntityType, IFoodChainAgent],
        n_agents: dict[str, int],
        width: int,
        height: int,
        landmarks: list[Entity] = None,
    ):
        self.food_chain = food_chain
        self.agents_per_type = n_agents
        self.width = width
        self.height = height
        self.landmarks = landmarks if landmarks is not None else []
        self.damping = 0.95

        self.n_species = len(self.food_chain)
        self.n_agents = sum(self.agents_per_type.values())
        self.n_landmarks = len(self.landmarks)

        self.observation_space_by_type = self._create_observation_space()
        self.action_space_by_type = self._create_action_space()
        self.observation_space = []
        self.action_space = []
        self.agents = self._create_agents()

        self.species_distances = {}
        self.distances = {}
        self.caught = defaultdict(int)

        self.mode = WorldType.RECTANGLE

    def _create_observation_space(self) -> spaces:
        space = spaces.Box(low=-1, high=1, shape=(2 * (self.n_agents + self.n_landmarks) + 2,), dtype=np.float32)
        return {agent_type: space for agent_type in self.food_chain.keys()}

    def _create_action_space(self) -> spaces:
        return {
            agent_type: spaces.Box(low=-1, high=+1, shape=(2, ), dtype=np.float32)
            for agent_type, agent in self.food_chain.items()
        }

    def _create_agents(self) -> list[BaseAgent]:
        agents = []
        for agent_type, n in self.agents_per_type.items():
            for i in range(n):
                agent = self.food_chain[agent_type]
                geometry = Geometry(
                    Shape.CIRCLE, color=agent.color, x=0, y=0, radius=agent.size
                )
                observation_space = self.observation_space_by_type[agent_type]
                action_space = self.action_space_by_type[agent_type]
                agents.append(
                    BaseAgent(
                        f"{agent_type}_{i}",
                        agent.type,
                        geometry=geometry,
                        observation_space=observation_space,
                        action_space=action_space,
                        preys=agent.preys,
                        predators=agent.predators,
                    )
                )
                self.observation_space.append(observation_space)
                self.action_space.append(action_space)
        return agents

    def reset(self) -> tuple[np.ndarray, dict]:
        for agent in self.agents:
            agent.set_position(np.random.uniform(0, self.width), np.random.uniform(0, self.height))
        # for landmark in self.landmarks:
        #     landmark.set_position(landmark.x, landmark.y)
        #     landmark.geometry.set_position(landmark.x, landmark.y)

    def observe(self, agent: BaseAgent) -> np.ndarray:
        relative_positions = []
        for other in self.agents:
            if other != agent:
                # print(f"Agent: {agent.type}, Other: {other.type}, Offset: {self._offset(agent, other, norm=True)}")
                relative_positions.extend(self._offset(agent, other, scaled=True))

        for landmark in self.landmarks:
            relative_positions.extend(self._offset(agent, landmark, scaled=True))

        relative_positions.extend([agent.x / self.width, agent.y / self.height])
        relative_positions.extend([agent.vx, agent.vy])
        return np.array(relative_positions)

    def reward(self, agent: BaseAgent) -> float:
        # Reward base on maximizing the distance to predators and minimizing the distance to preys
        reward = 0
        # Alpha and beta are hyperparameters that control the influence of hunting vs staying away
        alpha = 0.05
        beta = 0.05

        pred_types = self.food_chain[agent.type].predators
        prey_types = self.food_chain[agent.type].preys

        coop_distance = [dist[prey] for dist in self.species_distances[agent.type] for prey in prey_types]
        reward -= alpha * sum(coop_distance)

        # Reward for catching preys
        reward += 50 * self.caught[agent.type]

        # Maximize the distance to predators
        distances = self.distances[agent]
        for pred, distance in distances.items():
            if pred.type in pred_types:
                reward += beta * distance
                # If the predator is too close, penalize the agent
                radius = (agent.geometry.width / 2 + pred.geometry.width / 2) / self.width
                if distance < radius:
                    reward -= 10


        # Add border penalty
        for pos_coord in [agent.x / self.width, agent.y / self.width]:
            # Resize to be between -1 and 1
            pos_coord = 2 * pos_coord - 1
            if abs(pos_coord) >= 0.9:
                reward -= (abs(pos_coord) - 0.9) * 10

        # print(f"Agent: {agent.type}, Reward: {reward}")

        return reward

    def done(self, agent: BaseAgent) -> bool:
        # For now, we just terminate the episode withouth any condition
        return False

    def step(self):
        # import time

        # start = time.time()
        speed_factor = 1
        # Simply update the position of the agents based without any physics
        for agent in self.agents:
            agent.x += agent.vx * self.food_chain[agent.type].speed * speed_factor
            agent.y += agent.vy * self.food_chain[agent.type].speed * speed_factor

            # Torus world
            if self.mode == WorldType.TORUS:
                agent.x %= self.width
                agent.y %= self.height
            else:
                # Avoid agents to go outside the world
                agent.x = max(0, min(agent.x, self.width))
                agent.y = max(0, min(agent.y, self.height))

            for landmark in self.landmarks:
                if agent != landmark and landmark.can_collide:
                    if check_collision(agent, landmark, self.width, self.height, offset=0): # Change if torus
                        agent.x -= agent.vx * self.food_chain[agent.type].speed
                        agent.y -= agent.vy * self.food_chain[agent.type].speed

            agent.vx *= self.damping
            agent.vy *= self.damping

        # Fill the species_distances dict
        self.species_distances = {}
        for type in self.food_chain.keys():
            self.species_distances[type] = []
            # For each agent of this type, get the minimum distance to their preys for each prey type
            for agent in self.agents:
                if agent.type == type:
                    distances = {}
                    for prey_type in self.food_chain[type].preys:
                        prey_distances = []
                        for prey in self.agents:
                            if prey.type == prey_type:
                                prey_distances.append(self._distance(agent, prey, scaled=True))
                        distances[prey_type] = min(prey_distances)
                    self.species_distances[type].append(distances)

        # Fill the distances dict
        self.distances = {}
        self.caught = defaultdict(int)
        for agent in self.agents:
            self.distances[agent] = {}
            for other in self.agents:
                if agent != other:
                    self.distances[agent][other] = self._distance(agent, other, scaled=True)
                    radius = (agent.geometry.width / 2 + other.geometry.width / 2) / self.width
                    # if other.type in self.food_chain[agent.type].preys:
                    if self.distances[agent][other] < radius and other.type in self.food_chain[agent.type].preys:
                        # print(f"{agent.type} caught {other.type} with distance {self.distances[agent][other]} radius {radius}")
                        self.caught[agent.type] += 1

        # print(f"Step time: {time.time() - start}")
        # print(self.caught)
        # print(self.distances)
        # print(self.species_distances)