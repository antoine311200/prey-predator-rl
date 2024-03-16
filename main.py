import pyglet

from predator_prey.envs import MultiAgentEnvionment
from predator_prey.scenario.scenarios import get_scenarios

if __name__ == "__main__":
    scenario, instance = get_scenarios("simple_prey_predator")
    env = MultiAgentEnvionment(scenario)

    observations, info = env.reset()

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
