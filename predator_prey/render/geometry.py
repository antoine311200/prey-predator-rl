import pyglet

# Create Enum of shapes
from enum import Enum

class Shape(Enum):
    CIRCLE = 1
    SQUARE = 2
    TRIANGLE = 3
    RECTANGLE = 4

class Geometry:

    def __init__(self, shape: Shape, color: tuple, x: float = 0, y: float = 0, batch: pyglet.graphics.Batch = None, **kwargs):
        self.shape = shape
        self.color = color
        self.x = x
        self.y = y

        self.object = None
        if self.shape == Shape.CIRCLE:
            # Set width and height for collision detection
            radius = kwargs.get("radius", 10)
            self.width = radius * 2
            self.height = radius * 2

            self.center_point = pyglet.shapes.Circle(self.x, self.y, 2, color=(0, 0, 0), batch=batch)
            self.object = pyglet.shapes.Circle(self.x, self.y, radius, color=self.color)#, batch=batch)
            self.hit_box = pyglet.shapes.Box(self.x + self.width//2, self.y + self.height // 2, self.width, self.height, color=(0, 0, 0), batch=batch)
        elif self.shape == Shape.SQUARE:
            # Set width and height for collision detection
            size = kwargs.get("size", 10)
            self.width = size
            self.height = size

            self.object = pyglet.shapes.Rectangle(self.x, self.y, size, size, color=self.color, batch=batch)
            self.hit_box = pyglet.shapes.Box(self.x - self.width//2, self.y - self.height // 2, self.width, self.height, color=(0, 0, 0), batch=batch)
        elif self.shape == Shape.TRIANGLE:
            high = kwargs.get("high", 10)
            self.object = pyglet.shapes.Triangle(self.x, self.y, self.x + high, self.y, self.x + high/2, self.y + high, color=self.color, batch=batch)
            # Set width and height for collision detection
            self.width = high
            self.height = high
        elif self.shape == Shape.RECTANGLE:
            # Set width and height for collision detection
            width = kwargs.get("width", 10)
            height = kwargs.get("height", 10)
            self.width = width
            self.height = height

            # self.center_point = pyglet.shapes.Circle(10, 10, 2, color=(0, 0, 0), batch=batch)
            self.center_point = pyglet.shapes.Circle(self.x, self.y, 2, color=(0, 0, 0), batch=batch)
            self.object = pyglet.shapes.Rectangle(self.x - width//2, self.y - height//2, width, height, color=self.color, batch=batch)
            self.hit_box = pyglet.shapes.Box(self.x - width//2, self.y - height//2, self.width, self.height, color=(0, 0, 0), batch=batch)
        self.render_box = False

        # Text element
        self.label = pyglet.text.Label(
            text="",
            x=self.x,
            y=self.y+self.height//2*1.5,
            anchor_x="center",
            anchor_y="center",
            font_size=10,
            color=(0, 0, 0, 255),
            batch=batch
        )

    def set_position(self, x, y, render_box=False):
        self.x = x
        self.y = y
        self.center_point._x = x
        self.center_point._y = y

        if self.shape == Shape.CIRCLE:
            self.object._x = x
            self.object._y = y
        else:
            self.object._x = (x - self.width//2)
            self.object._y = (y - self.height//2)
        self.object._create_vertex_list()

        if render_box:
            self.center_point._create_vertex_list()

            self.hit_box._x = (x - self.width//2)
            self.hit_box._y = (y - self.height//2)
            self.hit_box._create_vertex_list()

            self.label.x = x
            self.label.y = y+self.height//2*2
            self.label.text = f"({x:.1f}, {y:.1f})"
            self.label._update()


    def render(self, render_box=False):
        self.object.draw()
        if render_box:
            self.hit_box.draw()
            self.center_point.draw()
            self.label.draw()