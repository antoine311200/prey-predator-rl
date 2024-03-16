import pyglet
from pyglet.gl import *

import numpy as np

# Class taken from OpenAI's gym
RAD2DEG = 57.29577951308232


class Instance:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.batch = pyglet.graphics.Batch()

        self.start()

        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def start(self):
        self.window = pyglet.window.Window(width=self.width, height=self.height)
        self.window.set_caption("Predator Prey")
        self.window.on_close = self.close


    def render(self, entities):
        glClearColor(1, 1, 1, 1)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        for entity in entities:
            entity.geometry.set_position(entity.x, entity.y)
            entity.geometry.render()

        # Draw a line to indicate the scale
        # pyglet.shapes.Line(10, 10, 50, 10, width=1, color=(0, 0, 0), batch=self.batch).draw()

        self.window.flip()

    def close(self):
        self.window.close()
