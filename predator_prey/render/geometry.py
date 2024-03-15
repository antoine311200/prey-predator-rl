import pyglet

# Create Enum of shapes
from enum import Enum

class Shape(Enum):
    CIRCLE = 1
    SQUARE = 2
    TRIANGLE = 3

class Geometry:

    def __init__(self, shape: Shape, color: tuple, x: float, y: float, radius: float, batch: pyglet.graphics.Batch = None):
        self.shape = shape
        self.color = color
        self.x = x
        self.y = y
        self.radius = radius

        self.object = None
        if self.shape == Shape.CIRCLE:
            self.object = pyglet.shapes.Circle(self.x, self.y, self.radius, color=self.color, batch=batch)
        elif self.shape == Shape.SQUARE:
            self.object = pyglet.shapes.Rectangle(self.x, self.y, self.radius, self.radius, color=self.color, batch=batch)
        elif self.shape == Shape.TRIANGLE:
            self.object = pyglet.shapes.Triangle(self.x, self.y, self.x + self.radius, self.y, self.x + self.radius/2, self.y + self.radius, color=self.color, batch=batch)

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.object._x = x
        self.object._y = y
        self.object._update_position()

    def render(self):
        self.object.draw()