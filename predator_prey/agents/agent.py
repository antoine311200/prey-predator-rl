import numpy as np
import torch
from torch import nn

from predator_prey.render.geometry import Geometry, Shape

# Create a EntityType string
from typing import NewType
EntityType = NewType("EntityType", str)

class Entity:
    def __init__(self, name, type: EntityType, x, y, geometry: Geometry = None, can_move: bool = True, can_collide: bool = True):
        self.name = name
        self.type = type

        self.x = x
        self.y = y

        self.vx = 0
        self.vy = 0

        self.can_move = can_move
        self.can_collide = can_collide

        if geometry is None:
            geometry = Geometry(Shape.CIRCLE, (0, 255, 0), x, y, 25)
        else:
            self.geometry = geometry

    def set_position(self, x, y):
        self.x = x
        self.y = y

class BaseAgent(Entity):

    def __init__(self, name, type: EntityType, x: float = 0, y: float = 0, communication: bool = False, geometry=None, **kwargs):
        super().__init__(name, type, x, y, geometry, True, True)
        # Each agent has a list of preys it can eat and a list of predators that can eat it
        self.preys: list[EntityType] = getattr(kwargs, "preys", [])
        self.predators: list[EntityType] = getattr(kwargs, "predators", [])

        self.action_space = None

        self.communication = communication


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