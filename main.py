from predator_prey.envs import MultiAgentEnvionment
from predator_prey.scenario import SimplePreyPredatorScenario
from predator_prey.render.render import Instance
from predator_prey.render.geometry import Geometry, Shape
from predator_prey.agents import BaseAgent, EntityType, Entity

import pyglet

if __name__ == "__main__":
    width, height = 800, 500
    instance = Instance(width, height)

    landmarks = [
        Entity(
            "landmark_0",
            EntityType("landmark"),
            x=width//2, y=height//2,
            geometry=Geometry(Shape.RECTANGLE, color=(0, 255, 0), x=width//2, y=height//2, width=250, height=15),
        ),
        Entity(
            "landmark_1",
            EntityType("landmark"),
            x=width//2, y=height//2,
            geometry=Geometry(Shape.RECTANGLE, color=(0, 255, 0), x=width//2, y=height//2, width=15, height=250),
        ),
    ]

    scenario = SimplePreyPredatorScenario(n_predators=8, n_preys=15, landmarks=landmarks)
    env = MultiAgentEnvionment(scenario)

    observations = env.reset(bounds=[width, height])

    step = 0
    max_steps = 1_000_000_000
    while step < max_steps:
        instance.render(scenario.entities)
        pyglet.clock.tick()
        if instance.window.has_exit:
            break

        # Take action and update environment
        actions = [agent.action_space.sample() for agent in env.agents]
        observations, rewards, dones, infos = env.step(actions)

        step += 1
