from predator_prey.envs import MultiAgentEnvionment
from predator_prey.scenario import SimplePreyPredatorScenario
from predator_prey.render.render import Instance

import pyglet

if __name__ == "__main__":
    width, height = 800, 500
    instance = Instance(width, height)

    scenario = SimplePreyPredatorScenario(n_predators=2, n_preys=5)
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