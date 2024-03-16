from copy import deepcopy
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        n_agents: int = 1,
        max_size: int = 1_000_000,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.max_size = max_size

        # Init buffers
        self.states = torch.zeros((max_size, n_agents, state_size), dtype=np.float32)
        self.actions = torch.zeros((max_size, n_agents, action_size), dtype=np.float32)
        self.reward = torch.zeros((max_size, n_agents, 1), dtype=np.float32)
        self.done = torch.zeros((max_size, n_agents, 1), dtype=np.float32)
        self.next_states = torch.zeros(
            (max_size, n_agents, state_size), dtype=np.float32
        )

        # Init pointers
        self.pointer = 0
        self.size = 0

    def clear(self) -> None:
        self.size = 0
        self.pointer = 0

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        self.states[self.pointer] = torch.FloatTensor(state)
        self.actions[self.pointer] = torch.FloatTensor(action)
        self.reward[self.pointer] = torch.FloatTensor(reward)
        self.done[self.pointer] = torch.FloatTensor(done)
        self.next_states[self.pointer] = torch.FloatTensor(next_state)

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(self.size)[:batch_size]

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.reward[indices]
        dones = self.done[indices]
        next_states = self.next_states[indices]

        return states, actions, rewards, dones, next_states


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self.mu, self.sigma
        )


class DDPG:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        actor_class,
        critic_class,
        device: str = "cpu",
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma

        # Init models
        self.ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros((action_size,)))
        self.actor = actor_class(state_size, action_size).to(device)
        self.critic = critic_class(state_size, action_size).to(device)

        self.target_actor = deepcopy(self.actor).to(device)
        self.target_critic = deepcopy(self.actor).to(device)

        # Init optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def act(self, state: np.ndarray, explore: bool = False) -> np.ndarray:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            noise = self.ou()
            noise = torch.FloatTensor(noise).unsqueeze(0).to(self.device)
            action = self.actor(state)

            if explore:
                action = action + noise
            action = action.clamp(-1, 1)

            action = action.cpu().numpy()[0]
            return action

    def get_current_action(self, states: torch.Tensor) -> torch.Tensor:
        return self.actor(states)

    def get_target_action(self, states: torch.Tensor) -> torch.Tensor:
        return self.target_actor(states)

    def train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        current_actions: torch.Tensor,
        target_actions: torch.Tensor,
        entropy: torch.Tensor,
    ) -> None:
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critic
        next_q = torch.where(dones, self.target_critic(next_states, target_actions), 0)
        y = rewards + self.gamma * next_q

        predicted_q = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(predicted_q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, current_actions).mean()
        actor_loss += entropy.mean() * 1e-3
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_actor_target(self) -> None:
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    def reset(self) -> None:
        self.ou.reset()


class MADDPG:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        actor_class: torch.nn.Module,
        critic_class: torch.nn.Module,
        n_agents: int = 1,
        max_buffer_size: int = int(1e6),
        gamma: float = 0.99,
        batch_size: int = 128,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rb = ReplayBuffer(state_size, action_size, n_agents, max_buffer_size)

        self.agents = [
            DDPG(state_size, action_size, gamma, actor_class, critic_class, self.device)
            for _ in range(n_agents)
        ]

        # Hyperparameters
        self.batch_size = batch_size
        self.update_target_every = 100

        # Init incrementing variables
        self.train_step = 0

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states: np.ndarray, explore: bool = False) -> np.ndarray:
        return np.array(
            [agent.act(state, explore) for agent, state in zip(self.agents, states)]
        )

    def remember(self, state, action, reward, done, next_state):
        self.rb.remember(state, action, reward, done, next_state)

    def train(self):
        self.train_step += 1
        # We don't update every step
        if (
            self.rb.size < self.batch_size
            or self.train_step % self.update_target_every != 0
        ):
            return
        states, actions, rewards, dones, next_states = self.rb.sample(self.batch_size)
        for incr, agent in enumerate(self.agents):
            # Get the actions of all agents
            current_action = torch.tensor(
                [
                    _agent.get_current_action(states[:, _incr])
                    for _incr, _agent in enumerate(self.agents)
                ]
            ).transpose(0, 1)
            # Get the target_actions, the batch size is first
            target_actions = torch.tensor(
                [
                    _agent.get_target_action(next_states[:, _incr])
                    for _incr, _agent in enumerate(self.agents)
                ]
            ).transpose(0, 1)
            agent.train(
                states[:, incr],
                actions,
                rewards[:, incr],
                next_states[:, incr],
                dones[:, incr],
                current_action,
                target_actions,
                entropy=(current_action[incr] ** 2),
            )

        # Update target networks
        for agent in self.agents:
            agent.update_actor_target()
