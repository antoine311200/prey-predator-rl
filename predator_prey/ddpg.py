import numpy as np

class ReplayBuffer:

    def __init__(self):
        pass

    def __len__(self):
        pass

    def add(self, state, action, reward, next_state, done):
        pass

    def clear(self):
        pass

    def sample(self, batch_size):
        pass


class DDPG:

    def __init__(self):
        pass

    def select_action(self, state):
        pass

    def train(self, batch_size):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass