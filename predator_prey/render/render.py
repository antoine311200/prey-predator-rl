import pyglet
from pyglet.gl import *

import numpy as np

# Class taken from OpenAI's gym
RAD2DEG = 57.29577951308232

class Transform:
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(
            self.translation[0], self.translation[1], 0
        )  # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


class Instance:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height)
        self.window.set_caption("Predator Prey")
        self.window.on_close = self.close

        self.batch = pyglet.graphics.Batch()
        self.transform = Transform()

        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def render(self, entities):
        glClearColor(1, 1, 1, 1)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self.transform.enable()

        for entity in entities:
            entity.geometry.set_position(entity.x, entity.y)
            entity.geometry.render()

        self.transform.disable()

        self.window.flip()


    def close(self):
        self.window.close()