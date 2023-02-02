import numpy as np
from scipy.spatial import distance_matrix

def polar_to_cartesian(polar, origin):
    """
    Convert polar to cartesian coordinates
    and shift to origin
    """
    x = polar[:,0] * np.cos(polar[:,1])
    y = polar[:,0] * np.sin(polar[:,1])
    return np.stack([y,x], axis=1) + origin
    

def cartesian_to_polar(cartesian, origin):
    """
    Shift cartesian coordinates by origin
    and convert to polar coordinates
    """
    cartesian = cartesian - origin
    r = np.sqrt(np.sum(cartesian ** 2, axis=1))
    theta = np.arctan2(cartesian[:,0], cartesian[:,1])
    return np.stack([r,theta], axis=1)

def get_particles_uniform_circle(n, max_r):
    """
    Get n uniformly distributed particles within a circle
    of radius max_r
    """
    phi = 2.0 * np.pi * np.random.rand(n) # Uniform sampling of phi
    r = max_r * np.sqrt(np.random.rand(n)) # Power sampling of r
    return np.stack([r,phi], axis=1)

def get_distance_matrix(particles):
    """
    Get the (Euclidean) distance matrix of all particles
    """
    return distance_matrix(particles, particles)

def get_repulsive_force(distance_matrix):
    """
    Compute the repulsive force between particles
    using the distance matrix
    """
    pass

def get_attractive_force(distance_matrix):
    """
    Compute the attractive force between particles
    using the distance matrix
    """
    pass


