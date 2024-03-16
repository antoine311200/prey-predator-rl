import pyglet

from predator_prey.agents import BaseAgent, Entity, EntityType
from predator_prey.envs import MultiAgentEnvionment
from predator_prey.render.geometry import Geometry, Shape
from predator_prey.render.render import Instance
from predator_prey.scenario import SimplePreyPredatorScenario

if __name__ == "__main__":
    width, height = 800, 500
    instance = Instance(width, height)

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

    scenario = SimplePreyPredatorScenario(
        n_predators=10, n_preys=10, landmarks=landmarks
    )
    env = MultiAgentEnvionment(scenario)

    observations, info = env.reset(bounds=[width, height])

    step = 0
    max_steps = 1_000_000_000
    while step < max_steps:
        instance.render(scenario.entities)
        pyglet.clock.tick()
        if instance.window.has_exit:
            break

        # Take action and update environment
        actions = [agent.action_space.sample() for agent in env.agents]
        observations, rewards, dones, truncated, infos = env.step(actions)

        step += 1
