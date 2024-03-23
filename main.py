import time

import numpy as np
import pyglet
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from predator_prey.ddpg import MADDPG
from predator_prey.envs import MultiAgentEnvionment
from predator_prey.models import Actor, Critic
from predator_prey.scenario.scenarios import get_scenarios


def push_scalar(writer, key, value, step):
    if writer is None:
        return
    writer.add_scalar(key, value, step)


def close_writer(writer):
    if writer is None:
        return
    writer.flush()
    writer.close()


if __name__ == "__main__":
    use_writer = False
    if use_writer:
        writer = SummaryWriter()
    else:
        writer = None

    max_steps = 50_000
    warmup_steps = 1000
    eval_every_n_episodes = 150

    # scenario, instance = get_scenarios("food_chain")
    scenario, instance = get_scenarios("food_chain", width=700, height=700)
    # scenario, instance = get_scenarios("big_prey_predators")
    env = MultiAgentEnvionment(scenario, n_steps=100)

    maddpg = MADDPG(
        env.state_size,
        env.action_size,
        hidden_size=64,
        actor_class=Actor,
        critic_class=Critic,
        n_agents=len(env.agents),
        warmup_steps=warmup_steps,
        train_every_n_steps=5,
    )

    obs, info = env.reset()


    n_episodes = 0
    step = 0

    cumul_train_reward = 0
    do_one_eval = False
    start = time.time()
    start_episode_time = time.time()
    while step < max_steps+1:
        # Take action and update environment
        actions = maddpg.act(obs, explore=True)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        cumul_train_reward += rewards[0]
        # Convert to numpy arrays for easier handling
        actions = np.array(actions)
        maddpg.remember(obs, actions, rewards, dones, next_obs)
        losses = maddpg.train()
        if losses is not None:
            for incr in range(len(env.agents)):
                for key, value in losses[incr].items():
                    push_scalar(writer, "losses/" + key, value, step)
        obs = next_obs
        if np.any(dones) or truncated:
            # Reset
            maddpg.reset()
            obs, info = env.reset()
            # Log
            push_scalar(writer, "train_reward", cumul_train_reward, step)
            push_scalar(writer, "time", time.time() - start, step)
            # Reset counters
            cumul_train_reward = 0
            n_episodes += 1
            if n_episodes % eval_every_n_episodes == 0:
                do_one_eval = True

            print(f"Episode {n_episodes} > {time.time() - start_episode_time:.2f}s")
            start_episode_time = time.time()

        if do_one_eval:
            obs, info = env.reset()
            # Init counters
            cumul_eval_reward = 0
            episode_len = 0
            while True:
                instance.render(scenario.entities, scenario.landmarks)
                pyglet.clock.tick()
                if instance.window.has_exit:
                    break
                # time.sleep(0.05)
                actions = maddpg.act(obs, explore=False)
                next_obs, rewards, dones, truncated, infos = env.step(actions)
                obs = next_obs
                cumul_eval_reward += rewards[0]
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
                    push_scalar(writer, "eval_reward", cumul_eval_reward, step)
                    push_scalar(writer, "eval_episode_length", episode_len, step)
                    maddpg.reset()
                    obs, info = env.reset()
                    break

        step += 1
        if step % 25_000 == 0:
            print("Saving model")
            maddpg.save("test")

    close_writer(writer)
    # Save render in tensorboard folder

    folder_to_save = writer.log_dir if writer is not None else "renders/"

    n_episode_render = 0
    incr_episode_render = 0
    incr_render = 0
    obs, info = env.reset()
    while True:
        # Render
        instance.render(scenario.entities, scenario.landmarks)
        if incr_episode_render < n_episode_render:
            instance.save_render(
                folder_to_save + f"/render_{incr_episode_render}_{incr_render}.png"
            )
        incr_render += 1
        pyglet.clock.tick()
        if instance.window.has_exit:
            break
        # time.sleep(0.07)
        actions = maddpg.act(obs, explore=False)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        # print("obs: ", obs[0][:2], "actions: ", actions[0], "rewards: ", rewards[0])
        print(" | ".join(f"{agent.type}: {reward:.3f}" for agent, reward in zip(env.agents, rewards)))
        obs = next_obs
        if np.any(dones) or truncated:
            maddpg.reset()
            obs, info = env.reset()
            incr_episode_render += 1
            incr_render = 0
