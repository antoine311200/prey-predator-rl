import time

import numpy as np
import pyglet
from matplotlib import pyplot as plt

from predator_prey.ddpg import MADDPG
from predator_prey.envs import MultiAgentEnvionment
from predator_prey.models import Actor, Critic
from predator_prey.scenario.scenarios import get_scenarios

if __name__ == "__main__":
    scenario, instance = get_scenarios("big_prey_predators")
    env = MultiAgentEnvionment(scenario, n_steps=100)

    maddpg = MADDPG(
        env.state_size,
        env.action_size,
        hidden_size=256,
        actor_class=Actor,
        critic_class=Critic,
        n_agents=len(env.agents),
    )
    maddpg.load("test")

    obs, info = env.reset()

    while True:
        pyglet.clock.tick()
        instance.render(scenario.entities, scenario.landmarks)
        if instance.window.has_exit:
            break
        time.sleep(0.05)
        actions = maddpg.act(obs, explore=False)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        print("obs: ", obs[0], "actions: ", actions[0], "rewards: ", rewards[0])
        obs = next_obs
        if np.any(dones) or truncated:
            maddpg.reset()
            obs, info = env.reset()
