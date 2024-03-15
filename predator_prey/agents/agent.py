import numpy as np
import torch
from torch import nn

from predator_prey.render.geometry import Geometry, Shape

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

        self.geometry: Geometry = None

    def set_position(self, x, y):
        self.x = x
        self.y = y

class BaseAgent(Entity):

    def __init__(self, name, type: AgentType, x: float = 0, y: float = 0, communication: bool = False, geometry=None, **kwargs):
        super().__init__(x, y)
        # Each agent has a list of preys it can eat and a list of predators that can eat it
        self.preys: list[AgentType] = getattr(kwargs, "preys", [])
        self.predators: list[AgentType] = getattr(kwargs, "predators", [])

        self.name = name
        self.type = type

        self.action_space = None

        self.can_move = True
        self.can_collide = True
        self.communication = communication

        if geometry is None:
            geometry = Geometry(Shape.CIRCLE, (0, 255, 0), x, y, 25)
        else:
            self.geometry = geometry

        # self.nn = nn.Sequential(
        #     nn.BatchNorm1d(),

    def step(self, action):
        if self.can_move:
            # print(f"Agent {self.name} moved to {action}")
            self.vx, self.vy = action[0]

            self.x += self.vx
            self.y += self.vy

    def __repr__(self):
        return f"Agent[{self.type} ({self.x}, {self.y})]"