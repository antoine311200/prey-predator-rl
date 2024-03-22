from typing import Tuple

from predator_prey.agents import Entity, EntityType
from predator_prey.render.geometry import Geometry, Shape
from predator_prey.render.render import Instance
from predator_prey.scenario import SimplePreyPredatorScenario
from predator_prey.scenario.base_scenario import BaseScenario
from predator_prey.scenario.food_chain_scenario import (
    SIMPLE_FOODCHAIN_RELATIONS,
    SIMPLE_LANDMARKS,
    FoodChainScenario,
)


def simple_prey_predator(width: int, height: int) -> BaseScenario:

    return SimplePreyPredatorScenario(
        n_predators=3, n_preys=1, landmarks=[], width=width, height=height
    )


def prey_predators(width: int, height: int) -> BaseScenario:

    return SimplePreyPredatorScenario(
        n_predators=3, n_preys=1, landmarks=[], width=width, height=height
    )


def big_prey_predators(width: int, height: int) -> BaseScenario:

    return SimplePreyPredatorScenario(
        n_predators=3,
        n_preys=1,
        landmarks=[],
        width=width,
        height=height,
        radius_predator=20,
    )


def food_chain(width: int, height: int) -> BaseScenario:
    return FoodChainScenario(
        food_chain=SIMPLE_FOODCHAIN_RELATIONS,
        # landmarks=SIMPLE_LANDMARKS,
        # n_agents={"low_agent": 10, "mid_agent": 5, "high_agent": 3, "super_agent": 1},
        # n_agents={"low_agent": 4, "mid_agent": 1, "high_agent": 2, "super_agent": 1},
        # n_agents={"target": 5, "low_agent": 1, "mid_agent": 0, "high_agent": 0, "super_agent": 0},
        # n_agents={"target": 5, "low_agent": 2, "mid_agent": 3, "high_agent": 0, "super_agent": 1},
        n_agents={
            "target": 1,
            "low_agent": 1,
            "mid_agent": 3,
            "high_agent": 0,
            "super_agent": 0,
        },
        # n_agents={"low_agent": 4, "mid_agent": 2, "high_agent": 0, "super_agent": 1},
        width=width,
        height=height,
    )


def get_scenarios(
    name: str, width: int = 500, height: int = 500
) -> Tuple[BaseScenario, Instance]:
    if name == "simple_prey_predator":
        instance = Instance(width, height)
        scenario = simple_prey_predator(width, height)
        return scenario, instance
    elif name == "prey_predators":
        instance = Instance(width, height)
        scenario = prey_predators(width, height)
        return scenario, instance
    elif name == "big_prey_predators":
        instance = Instance(width, height)
        scenario = big_prey_predators(width, height)
        return scenario, instance
    elif name == "very_big_prey_predators":
        instance = Instance(width, height)
        scenario = very_big_prey_predators(width, height)
        return scenario, instance
    elif name == "food_chain":
        instance = Instance(width, height, food_chain=SIMPLE_FOODCHAIN_RELATIONS)
        scenario = food_chain(width, height)
        return scenario, instance
    else:
        raise ValueError(f"Unknown scenario: {name}")
