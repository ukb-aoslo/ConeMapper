import numpy as np

def uniform_2D_sphere(n, max_r):
    """
    Get n uniformly distributed particles within a circle
    of radius max_r
    """
    phi = 2.0 * np.pi * np.random.rand(n) # Uniform sampling of phi
    r = max_r * np.sqrt(np.random.rand(n)) # Power sampling of r
    return np.stack([r,phi], axis=1)

def uniform_2D_rectangle(n, bbox_min : np.ndarray, bbox_max : np.ndarray):
    """
    Get n uniformly distributed particles within a rectangle
    defined by the bounding box (bbox_min, bbox_max)
    """
    return (np.zeros((n,2)) + bbox_min) + np.random.rand(n,2) * (bbox_max - bbox_min)