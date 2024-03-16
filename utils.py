import numpy as np


def torus_distance(entity1, entity2, width, height):
    dx = abs(entity1.x - entity2.x)
    dy = abs(entity1.y - entity2.y)
    dx = min(dx, width - dx)
    dy = min(dy, height - dy)
    return np.sqrt(dx ** 2 + dy ** 2)