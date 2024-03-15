import numpy as np
import torch
from torch import nn

# Create a AgentType string
from typing import NewType
AgentType = NewType("AgentType", str)

class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.vx = 0
        self.vy = 0

        self.can_move = True
        self.can_collide = True

        self.geometry = None

class BaseAgent(Entity):

    def __init__(self, name, type: AgentType, x: float = 0, y: float = 0, communication: bool = False, **kwargs):
        super().__init__(x, y)
        # Each agent has a list of preys it can eat and a list of predators that can eat it
        self.preys: list[AgentType] = getattr(kwargs, "preys", [])
        self.predators: list[AgentType] = getattr(kwargs, "predators", [])

        self.name = name
        self.type = type

        self.can_move = True
        self.can_collide = True
        self.communication = communication

        self.geometry = None # Need to implement geometry for rendering

        # self.nn = nn.Sequential(
        #     nn.BatchNorm1d(),
