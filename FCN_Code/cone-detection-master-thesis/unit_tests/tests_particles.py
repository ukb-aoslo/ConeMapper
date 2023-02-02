import numpy as np
import torch
import matplotlib.pyplot as plt

from components.particles import get_particles_uniform_circle, polar_to_cartesian, cartesian_to_polar

from utils.visualization import visualize_particles

def test_generation():
    particles = get_particles_uniform_circle(1000, 100)
    origin = np.array([73,-58]).T
    cartesian = polar_to_cartesian(particles, origin)
    visualize_particles(cartesian)
    return True

def test_conversion():
    particles = get_particles_uniform_circle(1000, 100)
    origin = np.array([73,-58]).T
    cartesian = cartesian_to_polar(particles, origin)
    polar = polar_to_cartesian(cartesian, origin)
    diff = np.sum(particles - polar)
    theta = 1e-7
    return diff < theta