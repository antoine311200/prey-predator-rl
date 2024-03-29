import time

import numpy as np
import pyglet
import tqdm
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


def push_histogram(writer, key, value, step):
    if writer is None:
        return
    writer.add_histogram(key, value, step)


def close_writer(writer):
    if writer is None:
        return
    writer.flush()
    writer.close()


MAX_STEPS = 400_000
N_STEPS = 100
EVAL_EVERY_N_EPISODES = 10
USE_WRITER = True

if __name__ == "__main__":
    if USE_WRITER:
        writer = SummaryWriter()
    else:
        writer = None
    scenario, instance = get_scenarios("big_prey_predators")
    env = MultiAgentEnvionment(scenario, n_steps=N_STEPS)

    maddpg = MADDPG(
        env.state_size,
        env.action_size,
        hidden_size=300,
        actor_class=Actor,
        critic_class=Critic,
        n_agents=len(env.agents),
        warmup_steps=10_000,
        train_every_n_steps=5,
    )

    obs, info = env.reset()

    n_episodes = 0
    step = 0
    cumul_train_reward = np.zeros(len(env.agents))
    do_one_eval = False
    start = time.time()
    pbar = tqdm.tqdm(total=MAX_STEPS)
    while pbar.n < MAX_STEPS:
        # Take action and update environment
        actions = maddpg.act(obs, explore=True)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        cumul_train_reward += rewards
        # Convert to numpy arrays for easier handling
        actions = np.array(actions)
        maddpg.remember(obs, actions, rewards, dones, next_obs)
        losses = maddpg.train()
        if losses is not None:
            for incr in range(len(env.agents)):
                for key, value in losses[incr].items():
                    push_scalar(writer, f"agent{incr}/" + key, value, step)
        obs = next_obs
        if np.any(dones) or truncated:
            # Reset
            maddpg.reset()
            obs, info = env.reset()
            # Log
            for incr in range(len(env.agents)):
                push_scalar(
                    writer, f"agent{incr}/train_reward", cumul_train_reward[incr], step
                )
            push_scalar(writer, "time", time.time() - start, step)
            push_scalar(writer, "is_caught", np.any(dones), step)
            # Reset counters
            cumul_train_reward = np.zeros(len(env.agents))
            n_episodes += 1
            if n_episodes % EVAL_EVERY_N_EPISODES == 0:
                do_one_eval = True

        if do_one_eval:
            obs, info = env.reset()
            # Init counters
            cumul_eval_reward = np.zeros(len(env.agents))
            all_actions = []
            episode_len = 0
            while True:
                # instance.render(scenario.entities, scenario.landmarks)
                # pyglet.clock.tick()
                # if instance.window.has_exit:
                #     break
                actions = maddpg.act(obs, explore=False)
                next_obs, rewards, dones, truncated, infos = env.step(actions)
                obs = next_obs
                cumul_eval_reward += rewards
                all_actions.append(actions)
                episode_len += 1
                if np.any(dones) or truncated:
                    do_one_eval = False
                    pbar.set_description(
                        f"Evaluation, Reward prey: {cumul_eval_reward[0]}, Episode length: {episode_len}, Step: {step}"
                    )
                    push_scalar(writer, "is_caught_eval", np.any(dones), step)
                    # Log
                    all_actions = np.array(all_actions[:episode_len])
                    for incr in range(len(env.agents)):
                        push_scalar(
                            writer,
                            f"agent{incr}/eval_reward",
                            cumul_eval_reward[incr],
                            step,
                        )
                        for incr_a in range(env.action_size):
                            push_histogram(
                                writer,
                                f"agent{incr}/eval_action_{incr_a}",
                                all_actions[:, incr, incr_a],
                                step,
                            )
                    push_scalar(writer, "eval_episode_length", episode_len, step)
                    maddpg.reset()
                    obs, info = env.reset()
                    break

        pbar.update()
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
        time.sleep(0.07)
        actions = maddpg.act(obs, explore=False)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        print("obs: ", obs[0][:2], "actions: ", actions[0], "rewards: ", rewards[0])
        obs = next_obs
        if np.any(dones) or truncated:
            maddpg.reset()
            obs, info = env.reset()
            incr_episode_render += 1
            incr_render = 0
