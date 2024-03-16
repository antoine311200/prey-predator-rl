from typing import Tuple

from predator_prey.agents import Entity, EntityType
from predator_prey.render.geometry import Geometry, Shape
from predator_prey.render.render import Instance
from predator_prey.scenario import SimplePreyPredatorScenario
from predator_prey.scenario.base_scenario import BaseScenario


def simple_prey_predator(width: int, height: int) -> BaseScenario:

    return SimplePreyPredatorScenario(
        n_predators=3, n_preys=1, landmarks=[], width=width, height=height
    )


def get_scenarios(
    name: str, width: int = 800, height: int = 500
) -> Tuple[BaseScenario, Instance]:
    if name == "simple_prey_predator":
        instance = Instance(width, height)
        scenario = simple_prey_predator(width, height)
        return scenario, instance
    else:
        raise ValueError(f"Unknown scenario: {name}")
