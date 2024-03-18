import numpy as np

def torus_offset(entity1, entity2, width, height):
    dx = abs(entity1.x - entity2.x)
    dy = abs(entity1.y - entity2.y)
    # Wrap around horizontally
    if dx > width / 2:
        dx = width - dx
    # Wrap around vertically
    if dy > height / 2:
        dy = height - dy
    return dx, dy

def torus_distance(entity1, entity2, width, height):
    dx, dy = torus_offset(entity1, entity2, width, height)
    return np.sqrt(dx ** 2 + dy ** 2)