import pyglet
from pyglet.gl import *

import numpy as np

from utils import torus_distance, torus_offset

# Class taken from OpenAI's gym
RAD2DEG = 57.29577951308232


class Instance:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.batch = pyglet.graphics.Batch()
        self.keys = pyglet.window.key.KeyStateHandler()

        self.start()

        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.RENDER_HIT_BOX = False
        self.RENDER_GRID = False

    def start(self):
        self.window = pyglet.window.Window(width=self.width, height=self.height)
        self.window.set_caption("Predator Prey")
        self.window.on_close = self.close

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.H:
                self.RENDER_HIT_BOX = not self.RENDER_HIT_BOX
                print("Render hit box:", self.RENDER_HIT_BOX, self)
            if symbol == pyglet.window.key.G:
                self.RENDER_GRID = not self.RENDER_GRID
                print("Render grid:", self.RENDER_GRID, self)

    def render(self, entities):
        glClearColor(1, 1, 1, 1)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.push_handlers(self.keys)

        # Draw grid
        if self.RENDER_GRID:
            for i in range(0, self.width, 25):
                pyglet.shapes.Line(i, 0, i, self.height, width=1, color=(200, 200, 200), batch=self.batch).draw()
            for i in range(0, self.height, 25):
                pyglet.shapes.Line(0, i, self.width, i, width=1, color=(200, 200, 200), batch=self.batch).draw()

        for entity in entities:
            entity.geometry.set_position(entity.x, entity.y)
            entity.geometry.render(self.RENDER_HIT_BOX)

        # Draw line between predator to prey
        if self.RENDER_HIT_BOX:
            for entity1 in entities:
                if entity1.type == "predator":
                    for entity2 in entities:
                        if entity2.type == "prey":
                            pyglet.shapes.Line(entity1.x, entity1.y, entity2.x, entity2.y, width=1, color=(0, 0, 0), batch=self.batch).draw()
                            # Add distance between predator and prey as text
                            distance = torus_distance(entity1, entity2, self.width, self.height)
                            offset = torus_offset(entity1, entity2, self.width, self.height)
                            rot_angle = np.arctan(offset[1] / offset[0]) * RAD2DEG
                            pyglet.text.Label(
                                text=f"{distance:.2f}",
                                x=(entity1.x + entity2.x) / 2,
                                y=(entity1.y + entity2.y) / 2,
                                anchor_x="center",
                                anchor_y="center",
                                rotation=rot_angle,
                                font_size=8,
                                color=(255, 0, 0, 255),
                                batch=self.batch
                            ).draw()

        self.window.flip()

    def close(self):
        self.window.close()
