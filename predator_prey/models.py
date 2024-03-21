import torch
import torch.nn.functional as F


class Actor(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_size: int, action_dim: int):
        super(Actor, self).__init__()

        self.l1 = torch.nn.Linear(state_dim, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, action_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.tanh(self.l3(a))


class Critic(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_size: int, action_dim: int):
        super(Critic, self).__init__()
        self.l1 = torch.nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Flatten the actions
        state = torch.flatten(state, start_dim=1)
        action = torch.flatten(action, start_dim=1)
        inputs = torch.cat([state, action], 1)
        q = F.relu(self.l1(inputs))
        q = F.relu(self.l2(q))
        return self.l3(q)