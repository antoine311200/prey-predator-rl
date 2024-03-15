import pyglet

# Create Enum of shapes
from enum import Enum

class Shape(Enum):
    CIRCLE = 1
    SQUARE = 2
    TRIANGLE = 3
    RECTANGLE = 4

class Geometry:

    def __init__(self, shape: Shape, color: tuple, x: float, y: float, batch: pyglet.graphics.Batch = None, **kwargs):
        self.shape = shape
        self.color = color
        self.x = x
        self.y = y

        self.object = None
        if self.shape == Shape.CIRCLE:
            radius = kwargs.get("radius", 10)
            self.object = pyglet.shapes.Circle(self.x, self.y, radius, color=self.color, batch=batch)
        elif self.shape == Shape.SQUARE:
            size = kwargs.get("size", 10)
            self.object = pyglet.shapes.Rectangle(self.x, self.y, size, size, color=self.color, batch=batch)
        elif self.shape == Shape.TRIANGLE:
            high = kwargs.get("high", 10)
            self.object = pyglet.shapes.Triangle(self.x, self.y, self.x + high, self.y, self.x + high/2, self.y + high, color=self.color, batch=batch)
        elif self.shape == Shape.RECTANGLE:
            width = kwargs.get("width", 10)
            height = kwargs.get("height", 10)
            self.object = pyglet.shapes.Rectangle(self.x, self.y, width, height, color=self.color, batch=batch)

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.object._x = x
        self.object._y = y
        self.object._update_position()

    def render(self):
        self.object.draw()