import time

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

    print(
        env.state_size,
        env.action_size,
    )

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
    max_steps = 10_000
    all_rewards = []
    cumul_reward = 0
    while step < max_steps:
        # Take action and update environment
        actions = agent.act(obs, explore=True)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        cumul_reward += rewards[1]
        # Convert to numpy arrays for easier handling
        actions = np.array(actions)
        agent.remember(obs, actions, rewards, dones, next_obs)
        agent.train()
        obs = next_obs
        if np.any(dones) or truncated:
            print("Resetting environment, Reward:", cumul_reward, "Step:", step)
            # Reset
            agent.reset()
            obs, info = env.reset()
            all_rewards.append(cumul_reward)
            cumul_reward = 0

        step += 1
        if step % 25_000 == 0:
            print("Saving model")
            agent.save("test")

    obs, info = env.reset()
    while True:
        instance.render(scenario.entities)
        pyglet.clock.tick()
        if instance.window.has_exit:
            break
        time.sleep(0.1)
        actions = agent.act(obs, explore=False)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        print("obs: ", obs[1][:1], "actions: ", actions[1], "rewards: ", rewards[1])
        obs = next_obs
        if np.any(dones) or truncated:
            obs, info = env.reset()
