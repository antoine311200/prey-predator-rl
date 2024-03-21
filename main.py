import time

import numpy as np
import pyglet
from matplotlib import pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

from predator_prey.ddpg import MADDPG
from predator_prey.envs import MultiAgentEnvionment
from predator_prey.models import Actor, Critic
from predator_prey.scenario.scenarios import get_scenarios

if __name__ == "__main__":
    # scenario, instance = get_scenarios("food_chain")
    # writer = SummaryWriter()
    # scenario, instance = get_scenarios("prey_predators", width=500, height=500)
    scenario, instance = get_scenarios("food_chain", width=400, height=400)
    env = MultiAgentEnvionment(scenario, n_steps=100)

    maddpg = MADDPG(
        env.state_size,
        env.action_size,
        hidden_size=64,
        actor_class=Actor,
        critic_class=Critic,
        n_agents=len(env.agents),
    )

    obs, info = env.reset()

    step = 0
    max_steps = 50_000
    eval_every_n_episodes = 10
    n_episodes = 0

    cumul_train_reward = 0
    do_one_eval = False
    start = time.time()
    while step < max_steps:
        # Take action and update environment
        actions = maddpg.act(obs, explore=True)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        cumul_train_reward += rewards[1]
        # Convert to numpy arrays for easier handling
        actions = np.array(actions)
        maddpg.remember(obs, actions, rewards, dones, next_obs)
        losses = maddpg.train()
        # if losses is not None:
        #     for incr in range(len(env.agents)):
        #         for key, value in losses[incr].items():
        #             writer.add_scalar("losses/" + key, value, step)
        obs = next_obs
        if np.any(dones) or truncated:
            # Reset
            maddpg.reset()
            obs, info = env.reset()
            # Log
            # writer.add_scalar("train_reward", cumul_train_reward, step)
            # writer.add_scalar("time", time.time() - start, step)
            # Reset counters
            cumul_train_reward = 0
            n_episodes += 1
            if n_episodes % eval_every_n_episodes == 0:
                do_one_eval = True

        if do_one_eval:
            obs, info = env.reset()
            # Init counters
            cumul_eval_reward = 0
            episode_len = 0
            while True:
                actions = maddpg.act(obs, explore=False)
                next_obs, rewards, dones, truncated, infos = env.step(actions)
                obs = next_obs
                cumul_eval_reward += rewards[1]
                episode_len += 1
                if np.any(dones) or truncated:
                    do_one_eval = False
                    print(
                        "Evaluation, Reward:",
                        cumul_eval_reward,
                        "Episode length:",
                        episode_len,
                        "Step:",
                        step,
                    )
                    # Log
                    # writer.add_scalar("eval_reward", cumul_eval_reward, step)
                    # writer.add_scalar("eval_episode_length", episode_len, step)
                    maddpg.reset()
                    obs, info = env.reset()
                    break

        step += 1
        if step % 25_000 == 0:
            print("Saving model")
            maddpg.save("test")

    # writer.flush()
    # writer.close()
    # Save render in tensorboard folder
    # folder_to_save = writer.log_dir

    n_episode_render = 3
    incr_episode_render = 0
    incr_render = 0
    obs, info = env.reset()
    while True:
        # Render
        instance.render(scenario.entities, scenario.landmarks)
        # if incr_episode_render < n_episode_render:
        #     instance.save_render(
        #         folder_to_save + f"/render_{incr_episode_render}_{incr_render}.png"
        #     )
        incr_render += 1
        pyglet.clock.tick()
        if instance.window.has_exit:
            break
        time.sleep(0.05)
        actions = maddpg.act(obs, explore=False)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        print("obs: ", obs[1][:2], "actions: ", actions[1], "rewards: ", rewards[1])
        obs = next_obs
        if np.any(dones) or truncated:
            maddpg.reset()
            obs, info = env.reset()
            incr_episode_render += 1
            incr_render = 0
