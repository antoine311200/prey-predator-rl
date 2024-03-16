import numpy as np

def torus_offset(entity1, entity2, width, height):
    dx = entity1.x - entity2.x
    dy = entity1.y - entity2.y
    dx = min(dx, width - dx)
    dy = min(dy, height - dy)
    return dx, dy

def torus_distance(entity1, entity2, width, height):
    dx, dy = torus_offset(entity1, entity2, width, height)
    return np.sqrt(dx ** 2 + dy ** 2)