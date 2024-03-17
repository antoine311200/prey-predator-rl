import numpy as np


def torus_offset(entity1, entity2, width, height, normalized=False):
    dx = entity1.x - entity2.x
    dy = entity1.y - entity2.y
    # Wrap around horizontally
    if abs(dx) > width / 2:
        dx = width - dx
    # Wrap around vertically
    if abs(dy) > height / 2:
        dy = height - dy
    if normalized:
        dx = dx / (width / 2)
        dy = dy / (height / 2)
    return dx, dy


def torus_distance(entity1, entity2, width, height, normalized=False):
    dx, dy = torus_offset(entity1, entity2, width, height, normalized=normalized)
    return np.sqrt(dx**2 + dy**2)
