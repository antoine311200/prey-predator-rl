from typing import Tuple

from predator_prey.agents import Entity, EntityType
from predator_prey.render.geometry import Geometry, Shape
from predator_prey.render.render import Instance
from predator_prey.scenario import SimplePreyPredatorScenario
from predator_prey.scenario.base_scenario import BaseScenario


def simple_prey_predator(width: int, height: int) -> BaseScenario:
    landmarks = [
        Entity(
            "landmark_0",
            EntityType("landmark"),
            x=width // 2,
            y=height // 2 + 50,
            geometry=Geometry(Shape.RECTANGLE, color=(0, 255, 0), width=600, height=15),
        ),
        Entity(
            "landmark_1",
            EntityType("landmark"),
            x=width // 2,
            y=height // 2 - 50,
            geometry=Geometry(Shape.RECTANGLE, color=(0, 255, 0), width=600, height=15),
        ),
        Entity(
            "landmark_2",
            EntityType("landmark"),
            x=width // 2 + 50,
            y=height // 2,
            geometry=Geometry(Shape.RECTANGLE, color=(0, 255, 0), width=15, height=600),
        ),
        Entity(
            "landmark_3",
            EntityType("landmark"),
            x=width // 2 - 50,
            y=height // 2,
            geometry=Geometry(Shape.RECTANGLE, color=(0, 255, 0), width=15, height=600),
        ),
    ]

    return SimplePreyPredatorScenario(
        n_predators=10, n_preys=10, landmarks=landmarks, width=width, height=height
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
