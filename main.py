import numpy as np
import pyglet

from predator_prey.ddpg import MADDPG
from predator_prey.envs import MultiAgentEnvionment
from predator_prey.models import Actor, Critic
from predator_prey.scenario.scenarios import get_scenarios

if __name__ == "__main__":
    # scenario, instance = get_scenarios("food_chain")
    scenario, instance = get_scenarios("simple_prey_predator")
    env = MultiAgentEnvionment(scenario, n_steps=1000)

    agent = MADDPG(
        env.state_size,
        env.action_size,
        hidden_size=64,
        actor_class=Actor,
        critic_class=Critic,
        n_agents=len(env.agents),
    )

    obs, info = env.reset()

    step = 0
    max_steps = 1_000_000_000
    while step < max_steps:
        if step > 0:
            instance.render(scenario.entities)
            pyglet.clock.tick()
            if instance.window.has_exit:
                break

        # Take action and update environment
        actions = agent.act(obs, explore=True)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        # Convert to numpy arrays for easier handling
        actions = np.array(actions)
        agent.remember(obs, actions, rewards, dones, next_obs)
        agent.train()
        obs = next_obs
        if np.any(dones) or truncated:
            print("Resetting environment")
            obs, info = env.reset()

        step += 1
        print("Step:", step)#, obs[0][0], dones[0])

        if step % 25_000 == 0:
            print("Saving model")
            agent.save("test")
